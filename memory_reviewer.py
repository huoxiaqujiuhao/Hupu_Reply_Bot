"""
memory_reviewer.py — 反思流水线
每次 main.py 启动时处理所有 reflected=0 且已过冷静期的 BotComments
"""
import sqlite3
import json
import time
import re
from openai import OpenAI
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
from config import CONFIG, get_logger
import memory_store

logger = get_logger("MemoryReviewer")

POSITIVE_THRESHOLD = CONFIG["memory_positive_threshold"]
NEGATIVE_THRESHOLD = CONFIG["memory_negative_threshold"]
COOLDOWN_HOURS     = CONFIG["reflect_cooldown_hours"]


# ══════════════════════════════════════════════
#  数据库迁移（首次运行时加列）
# ══════════════════════════════════════════════
def _migrate_db():
    conn = sqlite3.connect(CONFIG["db_name"])
    for ddl in [
        "ALTER TABLE BotComments ADD COLUMN reflected    INTEGER DEFAULT 0",
        "ALTER TABLE BotComments ADD COLUMN post_deleted INTEGER DEFAULT 0",
    ]:
        try:
            conn.execute(ddl)
        except Exception:
            pass
    conn.commit()
    conn.close()
    memory_store.init_db()


# ══════════════════════════════════════════════
#  LLM 反思 Prompt
# ══════════════════════════════════════════════
def _reflect_positive(title: str, post_content: str,
                      bot_reply: str, light_count: int,
                      llm: OpenAI) -> str | None:
    prompt = (
        f"你在虎扑帖子「{title}」下发表了评论，获得了 {light_count} 个赞。\n"
        f"帖子正文摘要：{post_content[:200]}\n"
        f"你的评论：「{bot_reply}」\n\n"
        "分析你为什么获得高赞，提炼 1 条行为准则。\n"
        "要求：必须包含【触发条件】+【具体手法】，不能只是抽象概括，15-30字。\n"
        '只输出 JSON：{"rule": "..."}'
    )
    try:
        resp = llm.chat.completions.create(
            model=CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=CONFIG["llm_timeout"],
        )
        return json.loads(resp.choices[0].message.content).get("rule")
    except Exception as e:
        logger.error(f"正样本反思失败: {e}")
        return None


def _reflect_dead_post(title: str, post_content: str,
                       total_replies: int, llm: OpenAI) -> str | None:
    prompt = (
        f"你在虎扑帖子「{title}」下发表了评论，但整个帖子只有 {total_replies} 条评论，完全没有热度。\n"
        f"帖子正文摘要：{post_content[:200]}\n\n"
        "分析这个帖子为什么没人讨论，提炼 1 条筛帖规则，帮助以后识别并避免这类冷帖。\n"
        "要求：聚焦帖子本身的特征（标题写法/话题类型/缺乏冲突等），15-30字。\n"
        '只输出 JSON：{"rule": "..."}'
    )
    try:
        resp = llm.chat.completions.create(
            model=CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=CONFIG["llm_timeout"],
        )
        return json.loads(resp.choices[0].message.content).get("rule")
    except Exception as e:
        logger.error(f"死帖反思失败: {e}")
        return None


def _reflect_wrong_direction(title: str, post_content: str,
                              bot_reply: str, light_count: int,
                              top_comments: list[dict],
                              llm: OpenAI) -> str | None:
    comments_str = "\n".join([
        f"{i+1}. 「{c['content'][:80]}」({c['lights']} 赞)"
        for i, c in enumerate(top_comments[:3])
    ])
    prompt = (
        f"帖子「{title}」下你的评论「{bot_reply}」只获得了 {light_count} 个赞。\n"
        f"帖子正文摘要：{post_content[:200]}\n\n"
        f"同帖高赞评论：\n{comments_str}\n\n"
        "对比高赞评论，分析你失败的原因，提炼 1 条改进规则，15-30字。\n"
        '只输出 JSON：{"rule": "..."}'
    )
    try:
        resp = llm.chat.completions.create(
            model=CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=CONFIG["llm_timeout"],
        )
        return json.loads(resp.choices[0].message.content).get("rule")
    except Exception as e:
        logger.error(f"方向错误反思失败: {e}")
        return None


# ══════════════════════════════════════════════
#  抓原帖（仅在没有 ReplyContext 时用）
# ══════════════════════════════════════════════
def _fetch_post_data(tid: int, context) -> dict | None:
    url = f"https://bbs.hupu.com/{tid}.html"
    try:
        resp = context.request.get(url, timeout=12000)
        if resp.status != 200:
            return None
        text = resp.text()
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', text, re.DOTALL)
        if not m:
            return None
        data   = json.loads(m.group(1))
        detail = data["props"]["pageProps"]["detail"]
        thread = detail["thread"]

        top_lights = []
        for light in detail.get("lights", [])[:5]:
            raw = light.get("content", "")
            clean = re.sub(r'<[^>]+>', '', raw).strip()
            if clean:
                top_lights.append({
                    "content": clean,
                    "lights":  light.get("allLightCount", 0),
                })
        return {
            "title":         thread.get("title", ""),
            "content":       re.sub(r'<[^>]+>', '', thread.get("content", "")).strip(),
            "total_replies": detail["replies"]["count"],
            "top_lights":    top_lights,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════
#  处理单条评论
# ══════════════════════════════════════════════
def _process_one(pid, tid, content, light_count,
                 emb_model, llm, context):
    # 中间地带直接跳过
    if NEGATIVE_THRESHOLD < light_count < POSITIVE_THRESHOLD:
        _mark_reflected(pid)
        return

    # 优先用发帖时预存的上下文
    ctx           = memory_store.get_reply_context(tid, content)
    post_title    = ctx["post_title"]         if ctx else ""
    post_content  = ctx["post_content"]       if ctx else ""
    post_category = ctx["post_category"]      if ctx else None
    post_total    = ctx["post_total_replies"] if ctx else None

    # 如果没有预存，实时抓帖
    live_data    = None
    post_deleted = False
    if post_total is None:
        live_data = _fetch_post_data(tid, context)
        if live_data is None:
            post_deleted = True
        else:
            post_total   = live_data["total_replies"]
            post_title   = post_title   or live_data["title"]
            post_content = post_content or live_data["content"]

    category = post_category or "global"

    # ── 正样本 ────────────────────────────────
    if light_count >= POSITIVE_THRESHOLD:
        title = post_title or f"tid={tid}"
        rule  = _reflect_positive(title, post_content, content, light_count, llm)
        if rule:
            result = memory_store.add_memory(
                rule_text=rule, category=category,
                rule_type="positive", evidence_likes=light_count,
                source_pid=pid, emb_model=emb_model,
            )
            logger.info(f"  ✅ 正样本 [{result}]: {rule[:50]}")

    # ── 负样本 ────────────────────────────────
    elif light_count <= NEGATIVE_THRESHOLD:
        if post_deleted:
            _mark_reflected(pid, post_deleted=True)
            return

        if post_total is not None and post_total < 10:
            # 死帖
            rule = _reflect_dead_post(post_title, post_content, post_total, llm)
            if rule:
                result = memory_store.add_memory(
                    rule_text=rule, category="filter",
                    rule_type="negative_dead", evidence_likes=light_count,
                    source_pid=pid, emb_model=emb_model,
                )
                logger.info(f"  ⚠️ 死帖规则 [{result}]: {rule[:50]}")
        else:
            # 活帖但方向错 — 需要高赞评论
            top_lights = (live_data or {}).get("top_lights", [])
            if not top_lights:
                fresh = _fetch_post_data(tid, context)
                if fresh:
                    top_lights = fresh.get("top_lights", [])

            if top_lights:
                rule = _reflect_wrong_direction(
                    post_title, post_content, content,
                    light_count, top_lights, llm,
                )
                if rule:
                    result = memory_store.add_memory(
                        rule_text=rule, category=category,
                        rule_type="negative_wrong", evidence_likes=light_count,
                        source_pid=pid, emb_model=emb_model,
                    )
                    logger.info(f"  ⚠️ 方向错规则 [{result}]: {rule[:50]}")

    _mark_reflected(pid)


def _mark_reflected(pid, post_deleted=False):
    conn = sqlite3.connect(CONFIG["db_name"])
    conn.execute(
        "UPDATE BotComments SET reflected=1, post_deleted=? WHERE pid=?",
        (1 if post_deleted else 0, pid)
    )
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════
def run(emb_model: SentenceTransformer, llm: OpenAI):
    _migrate_db()
    memory_store.decay_if_needed()

    cutoff = int(time.time()) - COOLDOWN_HOURS * 3600
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("""
        SELECT pid, tid, content, light_count
        FROM BotComments
        WHERE reflected=0 AND created_at < ?
        ORDER BY created_at ASC
    """, (cutoff,))
    pending = cur.fetchall()
    conn.close()

    if not pending:
        logger.info("没有待反思的评论，跳过")
        return

    logger.info(f"🧠 开始反思 {len(pending)} 条评论...")

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        for i, (pid, tid, content, light_count) in enumerate(pending):
            logger.info(f"  [{i+1}/{len(pending)}] pid={pid} light={light_count}")
            try:
                _process_one(pid, tid, content, light_count,
                             emb_model, llm, context)
            except Exception as e:
                logger.error(f"  处理 pid={pid} 出错: {e}")
            time.sleep(0.5)

    logger.info("✅ 反思完成")
    memory_store.print_stats()
