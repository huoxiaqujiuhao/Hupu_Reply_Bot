"""
memory_reviewer.py — 反思流水线（案例收集版）
每次 main.py 启动时，将历史 BotComments 转化为 Cases 案例存入数据库
不再用 LLM 提炼规则；直接存原始帖子 + 评论 + 点赞数
"""
import sqlite3
import json
import time
import re
from playwright.sync_api import sync_playwright
from embedder import EmbeddingModel
from openai import OpenAI
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
            txt = re.sub(r'<[^>]+>', '', light.get("content", "")).strip()
            if txt:
                top_lights.append({
                    "content":    txt,
                    "likes":      light.get("allLightCount", 0),
                    "created_at": light.get("createdAt", 0) // 1000,  # ms → s
                })

        return {
            "title":         thread.get("title", ""),
            "content":       re.sub(r'<[^>]+>', '', thread.get("content", "")).strip(),
            "total_replies": detail["replies"]["count"],
            "top_lights":    top_lights,
        }
    except Exception:
        return None


def _mark_reflected(pid, post_deleted=False):
    conn = sqlite3.connect(CONFIG["db_name"])
    conn.execute(
        "UPDATE BotComments SET reflected=1, post_deleted=? WHERE pid=?",
        (1 if post_deleted else 0, pid)
    )
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════
#  负样本诊断
# ══════════════════════════════════════════════
def _diagnose_negative(bot_reply: str, bot_ts: int, top_lights: list, emb_model) -> str:
    """
    对活帖低赞评论做因果诊断，返回 case_type：
      negative_timing    — bot 比最高赞晚入场，时机问题
      negative_duplicate — bot 比最高赞早，但内容高度相似（说了一样的话）
      negative_content   — bot 比最高赞早，内容不同，纯内容输了
    """
    if not top_lights:
        return "negative_content"

    best = max(top_lights, key=lambda x: x["likes"])
    best_ts = best["created_at"]

    # 时机判断：bot 比最高赞晚超过 5 分钟
    if best_ts > 0 and bot_ts > 0 and bot_ts > best_ts + 300:
        return "negative_timing"

    # 内容相似度判断（bot 早于或同期，但内容相近）
    try:
        import numpy as np
        vecs = emb_model.encode(
            [bot_reply, best["content"]],
            batch_size=2,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        sim = float(np.dot(vecs[0], vecs[1]))
        threshold = CONFIG.get("memory_duplicate_threshold", 0.82)
        if sim >= threshold:
            return "negative_duplicate"
    except Exception:
        pass

    return "negative_content"


# ══════════════════════════════════════════════
#  处理单条评论
# ══════════════════════════════════════════════
def _process_one(pid, tid, content, light_count, bot_ts, emb_model, context):
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

    # 如果没有预存，实时抓帖（同时需要 top_lights，所以都走实时路径）
    post_deleted = False
    top_lights   = []
    live_data = _fetch_post_data(tid, context)
    if live_data is None:
        if post_total is None:
            post_deleted = True
    else:
        post_total   = post_total   or live_data["total_replies"]
        post_title   = post_title   or live_data["title"]
        post_content = post_content or live_data["content"]
        top_lights   = live_data.get("top_lights", [])

    if post_deleted:
        _mark_reflected(pid, post_deleted=True)
        return

    category = post_category or "global"

    # ── 正样本 ────────────────────────────────
    if light_count >= POSITIVE_THRESHOLD:
        inserted = memory_store.add_case(
            tid=tid,
            post_title=post_title or f"tid={tid}",
            post_content=post_content,
            bot_reply=content,
            light_count=light_count,
            category=category,
            case_type="positive",
            emb_model=emb_model,
        )
        logger.info(f"  ✅ 正样本 {'[新增]' if inserted else '[已存在]'}: "
                    f"赞={light_count} | {(post_title or '')[:30]}")

    # ── 负样本 ────────────────────────────────
    elif light_count <= NEGATIVE_THRESHOLD:
        if post_total is not None and post_total < 10:
            # 死帖
            inserted = memory_store.add_case(
                tid=tid,
                post_title=post_title or f"tid={tid}",
                post_content=post_content,
                bot_reply=content,
                light_count=light_count,
                category=category,
                case_type="negative_dead",
                emb_model=emb_model,
            )
            logger.info(f"  ⚠️ 死帖案例 {'[新增]' if inserted else '[已存在]'}: "
                        f"总回复={post_total} | {(post_title or '')[:30]}")
        else:
            # 活帖低赞 — 诊断原因
            case_type = _diagnose_negative(content, bot_ts, top_lights, emb_model)
            best_light = max(top_lights, key=lambda x: x["likes"]) if top_lights else None
            diag_note  = ""
            if case_type == "negative_timing" and best_light:
                delay_min = max(0, (bot_ts - best_light["created_at"]) // 60)
                diag_note = f"晚{delay_min}min入场"
            elif case_type == "negative_duplicate" and best_light:
                diag_note = f"与高赞重复（{best_light['likes']}赞）"
            elif case_type == "negative_content":
                diag_note = "内容输了"

            inserted = memory_store.add_case(
                tid=tid,
                post_title=post_title or f"tid={tid}",
                post_content=post_content,
                bot_reply=content,
                light_count=light_count,
                category=category,
                case_type=case_type,
                emb_model=emb_model,
            )
            logger.info(f"  ⚠️ 负样本[{case_type}] {'[新增]' if inserted else '[已存在]'}: "
                        f"{diag_note} | 赞={light_count} | {(post_title or '')[:25]}")

    _mark_reflected(pid)


# ══════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════
def run(emb_model: EmbeddingModel, llm: OpenAI = None):
    _migrate_db()

    cutoff = int(time.time()) - COOLDOWN_HOURS * 3600
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("""
        SELECT pid, tid, content, light_count, created_at
        FROM BotComments
        WHERE reflected=0 AND created_at < ?
        ORDER BY created_at ASC
    """, (cutoff,))
    pending = cur.fetchall()
    conn.close()

    if not pending:
        logger.info("没有待处理的评论，跳过")
        return

    logger.info(f"🧠 开始收集案例，共 {len(pending)} 条评论...")

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        for i, (pid, tid, content, light_count, bot_ts) in enumerate(pending):
            logger.info(f"  [{i+1}/{len(pending)}] pid={pid} light={light_count}")
            try:
                _process_one(pid, tid, content, light_count, bot_ts, emb_model, context)
            except Exception as e:
                logger.error(f"  处理 pid={pid} 出错: {e}")
            time.sleep(0.5)

    logger.info("✅ 案例收集完成")
    memory_store.print_stats()
