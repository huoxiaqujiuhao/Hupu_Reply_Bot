"""
reply_bot.py — 回复机器人（消费端）

急行军逻辑（与原版不同）：
  急行军不是基于单轮的剩余时间，而是基于累计回复时间。
  is_panic_fn() 在每个候选帖子处理前动态判断，
  所以一轮内可能从从容模式自然过渡到急行军模式。

  ⚠️  classify_post() 仍然调用 LLM，这是用来给当前待回复帖子打标签，
      目的是找到数据库里同类的历史帖子作为 RAG 参考。
      这个 LLM 调用和数据库爬虫的分类是独立的，是必要的。
"""
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import sqlite3
import json
import time
import random
import re
import numpy as np
from config import CONFIG, get_logger
import memory_store

logger = get_logger("ReplyBot")


# ══════════════════════════════════════════════
#  去重记录
# ══════════════════════════════════════════════
def load_replied_history() -> set:
    try:
        with open(CONFIG["replied_history_file"], encoding="utf-8") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()

def save_replied_history(replied_urls: set):
    with open(CONFIG["replied_history_file"], "w", encoding="utf-8") as f:
        json.dump(list(replied_urls), f, ensure_ascii=False)


# ══════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════
def clean_html(raw: str) -> str:
    if not raw: return ""
    raw = re.sub(r'<img[^>]*>', '', raw)
    raw = re.sub(r'<[^>]+>', '', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    return raw.strip()

def get_next_data(page) -> dict | None:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=15000)
        page.wait_for_selector("#__NEXT_DATA__", state="attached", timeout=15000)
        return json.loads(page.locator("#__NEXT_DATA__").inner_text())
    except Exception as e:
        logger.warning(f"__NEXT_DATA__ 读取失败: {str(e)[:60]}")
        return None

def human_like_behavior(page):
    y = random.randint(*CONFIG["scroll_y_range"])
    page.evaluate(f"window.scrollTo(0, {y})")
    time.sleep(random.uniform(*CONFIG["scroll_pause_1"]))
    page.evaluate(f"window.scrollTo(0, {y + random.randint(100, 400)})")
    time.sleep(random.uniform(*CONFIG["scroll_pause_2"]))

def random_sleep(range_tuple: tuple, label="休息"):
    t = random.uniform(*range_tuple)
    logger.info(f"💤 {label} {t:.1f}s")
    time.sleep(t)

def parse_minutes_ago(time_str: str) -> int:
    if not time_str: return 99999
    if "刚刚" in time_str or "秒" in time_str: return 1
    m = re.search(r'(\d+)\s*分钟', time_str)
    if m: return int(m.group(1))
    m = re.search(r'(\d+)\s*小时', time_str)
    if m: return int(m.group(1)) * 60
    return 99999


# ══════════════════════════════════════════════
#  列表页扫描
# ══════════════════════════════════════════════
def collect_candidate_urls(page, label="") -> list:
    tag = f"[{label}] " if label else ""
    candidates, seen = [], set()
    for page_num in range(1, CONFIG["list_pages"] + 1):
        url = (CONFIG["list_url_base"] if page_num == 1
               else f"{CONFIG['list_url_base']}-{page_num}")
        logger.info(f"📋 {tag}列表第 {page_num} 页: {url}")
        try:
            page.goto(url, timeout=15000)
            human_like_behavior(page)
            for el in page.locator("a[href]").all():
                href = el.get_attribute("href")
                if href and re.match(r'^/\d+\.html$', href):
                    full_url = "https://bbs.hupu.com" + href
                    if full_url not in seen:
                        seen.add(full_url)
                        title = el.inner_text().strip().replace('\n', '')
                        if len(title) > 3:
                            candidates.append({"url": full_url, "title": title})
        except Exception as e:
            logger.error(f"扫描失败: {e}")
        if page_num < CONFIG["list_pages"]:
            random_sleep(CONFIG["reply_delay_pages"], "翻页间隔")
    logger.info(f"🎯 {tag}共发现 {len(candidates)} 个候选帖子")
    return candidates


# ══════════════════════════════════════════════
#  帖子解析
# ══════════════════════════════════════════════
def parse_post(page) -> dict | None:
    data = get_next_data(page)
    if not data: return None
    try:
        detail      = data["props"]["pageProps"]["detail"]
        thread      = detail["thread"]
        raw_content = thread.get("content", "")
        content     = clean_html(raw_content)
        has_media   = bool(re.search(r'<img|<video|<iframe', raw_content, re.IGNORECASE))

        if has_media and len(content) < 20:
            logger.info("纯图片/视频帖无文字，跳过")
            return None

        current_replies = []
        for rep in detail.get("replies", {}).get("list", [])[:CONFIG["current_replies_count"]]:
            txt = clean_html(rep.get("content", ""))
            if txt and "[图片]" not in txt:
                current_replies.append(txt)

        top_lights = []
        for light in detail.get("lights", [])[:CONFIG["reply_top_lights"]]:
            top_lights.append({
                "user":    light["author"]["puname"],
                "content": clean_html(light["content"]),
                "lights":  light["allLightCount"],
            })

        return {
            "url":             page.url,
            "title":           thread["title"],
            "post_time":       thread.get("createdAtFormat", ""),
            "total_replies":   detail["replies"]["count"],
            "content":         content,
            "top_lights":      top_lights,
            "current_replies": current_replies,
        }
    except KeyError:
        return None


# ══════════════════════════════════════════════
#  内存向量库
# ══════════════════════════════════════════════
class InMemoryVectorStore:
    def __init__(self, db_name: str, emb_model: SentenceTransformer):
        self.urls, self.titles, self.ai_tags = [], [], []
        self.matrix = None

        conn = sqlite3.connect(db_name)
        cur  = conn.cursor()
        cur.execute("""
            SELECT url, title, content, ai_tag FROM Posts
            WHERE ai_tag IS NOT NULL
              AND ai_tag NOT LIKE 'ERROR%'
              AND ai_tag != 'other'
        """)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            logger.warning("向量库：数据库中没有已打标帖子")
            self.matrix = np.zeros((0, 0), dtype=np.float32)
            return

        texts = []
        for url, title, content, ai_tag in rows:
            self.urls.append(url)
            self.titles.append(title)
            self.ai_tags.append(ai_tag)
            texts.append(f"{title}。{(content or '')[:CONFIG['post_text_max_chars']]}")

        vecs = emb_model.encode(
            texts,
            batch_size=CONFIG["vector_batch_size"],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.matrix = vecs.astype(np.float32)
        logger.info(f"向量库就绪：{len(self.urls)} 篇，维度 {self.matrix.shape[1]}")

    def query(self, vec: np.ndarray, ai_tag: str, top_k: int) -> list[dict]:
        if self.matrix.shape[0] == 0:
            return []
        mask = np.array([t == ai_tag for t in self.ai_tags])
        if not mask.any():
            return []
        sims         = self.matrix[mask] @ vec
        filtered_idx = np.where(mask)[0]
        top_local    = np.argsort(sims)[::-1][:top_k]
        return [
            {"url": self.urls[filtered_idx[i]], "title": self.titles[filtered_idx[i]],
             "similarity": float(sims[i])}
            for i in top_local
        ]


# ══════════════════════════════════════════════
#  RAG 流水线
# ══════════════════════════════════════════════
def classify_post(title: str, content: str, llm: OpenAI,
                  emb_model: SentenceTransformer = None) -> dict:
    """
    用 LLM 给当前待回复帖子打标签，用于 RAG 检索。
    注意：这里调用 LLM 是为了找到相似的历史帖子，不是批量打标。
    """
    try:
        with open(CONFIG["taxonomy_file"], encoding="utf-8") as f:
            taxonomy = json.load(f)
        taxonomy_str = json.dumps(taxonomy, ensure_ascii=False, separators=(',', ':'))

        filter_block = ""
        if emb_model is not None:
            query_text  = f"{title}。{(content or '')[:200]}"
            dead_cases  = memory_store.search_cases(
                query_text=query_text,
                case_type="negative_dead",
                emb_model=emb_model,
                top_k=CONFIG["memory_top_k_filter"],
            )
            if dead_cases:
                examples = "\n".join(
                    f"  {i+1}. 标题《{c['post_title'][:40]}》"
                    for i, c in enumerate(dead_cases)
                )
                filter_block = (
                    "\n【历史冷帖案例——以下类型的帖子你曾回复后几乎没有获赞，说明帖子本身缺乏讨论热度，打分时酌情降低】\n"
                    + examples
                )

        system_prompt = (
            CONFIG["prompt_classify_system"]
            + filter_block
            + '\n必须只输出严格的 JSON：{"primary_category":"...","secondary_tag":"...","discussion_value":8}'
        )
        resp = llm.chat.completions.create(
            model=CONFIG["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content":
                    f"菜单：{taxonomy_str}\n\n"
                    f"标题：{title}\n正文：{(content or '')[:CONFIG['post_prompt_max_chars']]}"},
            ],
            response_format={"type": "json_object"},
            temperature=CONFIG["classify_temperature"],
            timeout=CONFIG["llm_timeout"],
        )
        r = json.loads(resp.choices[0].message.content)
        return {
            "ai_tag":      f"{r.get('primary_category','')}-{r.get('secondary_tag','')}",
            "value_score": int(r.get("discussion_value", 0)),
        }
    except Exception as e:
        logger.warning(f"分类请求失败: {e}")
        return {"ai_tag": "未知类别", "value_score": 0}


def fetch_top_comments(urls: list[str]) -> list[dict]:
    conn, result = sqlite3.connect(CONFIG["db_name"]), []
    cur = conn.cursor()
    for url in urls:
        cur.execute("SELECT title FROM Posts WHERE url=?", (url,))
        row = cur.fetchone()
        if not row: continue
        cur.execute("""
            SELECT content, lights FROM Comments
            WHERE post_url=? AND lights>=?
              AND content IS NOT NULL AND content!='' AND content!='[图片]'
            ORDER BY lights DESC LIMIT ?
        """, (url, CONFIG["min_lights"], CONFIG["top_k_comments"]))
        for content, lights in cur.fetchall():
            result.append({"post_title": row[0], "comment": content, "lights": lights})
    conn.close()
    return result


def analyze_why_high_lights(comments: list[dict], llm: OpenAI) -> list[dict]:
    if not comments: return []
    batch = "\n".join([
        f"{i+1}. 帖子:《{c['post_title'][:20]}》 "
        f"评论:{c['comment'][:CONFIG['comment_ref_max_chars']]} 亮灯:{c['lights']}"
        for i, c in enumerate(comments)
    ])
    try:
        resp = llm.chat.completions.create(
            model=CONFIG["model"],
            messages=[
                {"role": "system", "content": CONFIG["prompt_analyze_system"]},
                {"role": "user", "content": batch},
            ],
            response_format={"type": "json_object"},
            temperature=CONFIG["analyze_temperature"],
            timeout=CONFIG["llm_timeout"],
        )
        parsed  = json.loads(resp.choices[0].message.content.strip())
        arr     = parsed if isinstance(parsed, list) else next(iter(parsed.values()))
        why_map = {item["id"]: item["why"] for item in arr}
        for i, c in enumerate(comments):
            c["why"] = why_map.get(i+1, "切中用户情绪共鸣点")
    except Exception:
        for c in comments:
            c["why"] = "切中用户情绪共鸣点"
    return comments


def rag_generate(
    title, content, ai_tag,
    vector_store: InMemoryVectorStore,
    emb_model: SentenceTransformer,
    llm: OpenAI,
    current_replies: list[str] = None,
) -> str:
    vec      = emb_model.encode(
        f"{title}。{(content or '')[:CONFIG['post_text_max_chars']]}",
        normalize_embeddings=True,
    )
    similar  = vector_store.query(vec, ai_tag, CONFIG["top_k_posts"])
    comments = fetch_top_comments([p["url"] for p in similar])
    if comments:
        comments = analyze_why_high_lights(comments, llm)

    if similar:
        logger.info(f"最相似帖：《{similar[0]['title'][:25]}》 相似度 {similar[0]['similarity']:.3f}")
    logger.info(f"参考高赞评论：{len(comments)} 条")

    ref_block = ("\n".join([
        f"  {i+1}. 「{c['comment'][:CONFIG['comment_ref_max_chars']]}」\n"
        f"     → 高赞原因：{c['why']}"
        for i, c in enumerate(comments)
    ]) if comments else "  （暂无高相似度历史评论）")

    current_vibe_block = (
        "\n".join([f"  - {r}" for r in current_replies])
        if current_replies else "  （目前暂无回复）"
    )

    pos_cases   = memory_store.search_cases(
        query_text=f"{title}。{(content or '')[:200]}",
        case_type="positive",
        emb_model=emb_model,
        category=ai_tag,
        top_k=CONFIG["memory_top_k_cases"],
    )
    memory_block = ""
    if pos_cases:
        case_lines = []
        for c in pos_cases:
            case_lines.append(
                f"- 帖子《{c['post_title'][:30]}》\n"
                f"  你当时的评论：「{c['bot_reply'][:100]}」（获得 {c['light_count']} 赞）"
            )
        memory_block = (
            "\n【你过去的成功评论案例（最相似话题，参考切入角度和语气，不要照抄）】\n"
            + "\n".join(case_lines)
        )

    system = CONFIG["prompt_generate_system"] + memory_block
    user = (
        f"【帖子类别】{ai_tag}\n"
        f"【标题】{title}\n"
        f"【正文】{(content or '')[:CONFIG['post_prompt_max_chars']]}\n\n"
        f"【当前评论风向（决定你的立场和情绪，必须顺势而为）】\n{current_vibe_block}\n\n"
        f"【历史同类高赞参考（只学语气节奏、黑话用法、断句习惯和大概评论结构和长度）】\n{ref_block}\n\n"
        "请结合当前气氛，直接输出你的评论内容（不要任何前缀和解释）："
    )
    resp = llm.chat.completions.create(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=CONFIG["generate_temperature"],
        max_tokens=CONFIG["max_tokens"],
        timeout=CONFIG["llm_timeout"],
    )
    return resp.choices[0].message.content.strip()


# ══════════════════════════════════════════════
#  真实发帖
# ══════════════════════════════════════════════
def real_reply_action(page, reply_text: str) -> bool:
    logger.info(f"📝 准备发送：{reply_text}")
    try:
        editor = page.locator(".ProseMirror").first
        if editor.count() == 0:
            logger.warning("未找到输入框，帖子可能已锁")
            return False

        editor.scroll_into_view_if_needed()
        time.sleep(random.uniform(*CONFIG["pre_click_pause"]))
        editor.click()
        time.sleep(random.uniform(*CONFIG["post_click_pause"]))

        editor.press_sequentially(reply_text, delay=CONFIG["typing_delay_ms"])
        time.sleep(random.uniform(*CONFIG["post_type_pause"]))

        send_selectors = [
            "div[class*='operatorButton']:has-text('回复')",
            "div[class*='operatorButtonContainer']:has-text('回复')",
            "div:text-is('回复')",
        ]
        for sel in send_selectors:
            btn = page.locator(sel).last
            if btn.count() > 0 and btn.is_visible():
                btn.scroll_into_view_if_needed()
                time.sleep(random.uniform(*CONFIG["post_click_pause"]))
                btn.click()
                logger.info(f"✅ 发送成功！")
                time.sleep(random.uniform(*CONFIG["post_send_pause"]))
                return True

        logger.warning("未找到发送按钮，放弃本次发送")
        try:
            editor.triple_click()
            editor.press("Backspace")
        except Exception:
            pass
        return False

    except Exception as e:
        logger.error(f"发帖出错: {e}")
        try:
            page.locator(".ProseMirror").first.fill("")
        except Exception:
            pass
        return False


# ══════════════════════════════════════════════
#  单轮次扫描 + 回复
# ══════════════════════════════════════════════
def run_scan_loop(
    page, candidates, replied_urls, replied_count,
    is_panic_fn,   # 可调用，动态判断当前是否急行军
    deadline,
    vector_store, emb_model, llm,
    low_score_urls=None,
) -> int:
    if low_score_urls is None:
        low_score_urls = set()

    new_replies = 0

    for i, candidate in enumerate(candidates):
        if time.time() >= deadline:
            logger.info("⏰ 回复时间到，停止")
            break
        if replied_count + new_replies >= CONFIG["max_reply_actions"]:
            logger.info("✅ 回复配额已满，收工")
            break

        url = candidate["url"]
        if url in replied_urls:
            continue

        # 每个帖子前动态判断当前模式
        panic      = is_panic_fn()
        mode_label = "【急行军】" if panic else "【从容】"

        if not panic and url in low_score_urls:
            logger.debug(f"{mode_label} 质检黑名单，跳过：{candidate.get('title','')[:25]}")
            continue

        logger.info(f"[{i+1}/{len(candidates)}] {mode_label} 进帖...")

        try:
            page.goto(url, timeout=15000)
            human_like_behavior(page)
            post_data = parse_post(page)
            if not post_data:
                continue

            rc          = post_data["total_replies"]
            minutes_ago = parse_minutes_ago(post_data["post_time"])
            logger.info(f"  《{post_data['title'][:25]}》 "
                        f"发布:{minutes_ago}min前 回复:{rc}")

            # 死帖判定
            if minutes_ago > CONFIG["dead_post_max_minutes"]:
                logger.info(f"  超过 {CONFIG['dead_post_max_minutes']} 分钟，死帖")
                continue
            if minutes_ago > CONFIG["dead_post_min_minutes"] and rc < CONFIG["dead_post_min_replies"]:
                logger.info("  发出超时且无人问津，放弃")
                continue

            # 触发判定（根据当前模式）
            need_reply = False
            status_msg = ""

            if not panic:
                if (CONFIG["calm_min_replies"] <= rc <= CONFIG["calm_max_replies"]
                        and minutes_ago <= CONFIG["calm_max_minutes"]):
                    status_msg = "🔥 极品坑位，AI 质检"
                    need_reply = True
                else:
                    status_msg = "⏳ 坑位条件不符，跳过"
            else:
                if minutes_ago < CONFIG["too_fresh_minutes"]:
                    status_msg = "⏳ 太新，让子弹飞"
                elif minutes_ago <= CONFIG["panic_golden_max"]:
                    status_msg = "🔥 黄金时段，直冲"
                    need_reply = True
                elif minutes_ago <= CONFIG["panic_scavenge_max"]:
                    if rc > CONFIG["panic_reply_cap"]:
                        status_msg = f"⏭️ 坑位已满（>{CONFIG['panic_reply_cap']}评）"
                    else:
                        status_msg = "🎯 捡漏，入场"
                        need_reply = True
                else:
                    status_msg = "⏭️ 超出时间窗口"

            logger.info(f"  {mode_label} {status_msg}")
            if not need_reply:
                continue

            # AI 质检（LLM 调用，用于 RAG 检索，见文件头部注释）
            post_analysis = classify_post(post_data["title"], post_data["content"], llm, emb_model)
            ai_tag        = post_analysis["ai_tag"]
            value_score   = post_analysis["value_score"]
            logger.info(f"  分类: {ai_tag} | 潜力值: {value_score}/10")

            if "未知" in ai_tag or "-" not in ai_tag or value_score == 0:
                logger.warning("  分类失败或疑似敏感，防封跳过")
                replied_urls.add(url)
                continue

            if not panic and value_score < CONFIG["calm_quality_min"]:
                logger.info(f"  质检未通过（{value_score} < {CONFIG['calm_quality_min']}）")
                low_score_urls.add(url)
                continue
            elif panic:
                logger.info("  急行军免检，强行发车")
            else:
                logger.info("  质检通过，准备开火")

            reply_text = rag_generate(
                post_data["title"], post_data["content"], ai_tag,
                vector_store, emb_model, llm,
                current_replies=post_data["current_replies"],
            )

            logger.info(f"\n  {'─'*50}")
            logger.info(f"  💬 生成回复：{reply_text}")
            logger.info(f"  {'─'*50}\n")

            success = real_reply_action(page, reply_text)
            if success:
                new_replies += 1
                # 发帖成功，记录上下文供日后反思
                try:
                    tid = int(re.search(r'/(\d+)\.html', url).group(1))
                    memory_store.save_reply_context(
                        tid=tid,
                        content=reply_text,
                        post_title=post_data["title"],
                        post_content=post_data.get("content", ""),
                        post_category=ai_tag,
                        post_total_replies=post_data["total_replies"],
                    )
                except Exception:
                    pass
            replied_urls.add(url)
            save_replied_history(replied_urls)

        except Exception as e:
            logger.error(f"抓取出错: {e}", exc_info=True)

        if i < len(candidates) - 1:
            random_sleep(CONFIG["reply_delay_posts"], "帖子间隔")

    return new_replies


# ══════════════════════════════════════════════
#  主控（被 main.py 调用）
# ══════════════════════════════════════════════
def auto_crawler(
    deadline: float,
    cum_reply_secs_at_start: float,
    total_reply_budget: float,
    vector_store: InMemoryVectorStore,
    emb_model: SentenceTransformer,
    llm: OpenAI,
):
    """
    deadline:                  本轮回复的截止时间戳
    cum_reply_secs_at_start:   进入本轮时已累计的回复秒数
    total_reply_budget:        整个 session 的总回复秒数预算
    """
    phase_start = time.time()
    replied_urls  = load_replied_history()
    replied_count = 0
    logger.info(f"已加载历史去重记录：{len(replied_urls)} 个 URL")

    # 动态急行军判断
    def is_panic_fn() -> bool:
        elapsed = time.time() - phase_start
        cum     = cum_reply_secs_at_start + elapsed
        if total_reply_budget <= 0:
            return False
        is_p = (cum / total_reply_budget) >= CONFIG["panic_ratio"]
        return is_p

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page    = context.pages[0]

        # ── 从容扫描（直到 deadline、配额满、或切入急行军）──────
        calm_round     = 1
        low_score_urls = set()

        while time.time() < deadline and replied_count < CONFIG["max_reply_actions"]:
            if is_panic_fn():
                break   # 进入急行军，退出从容循环

            logger.info(f"\n{'─'*50}")
            logger.info(f"😌 从容模式 第 {calm_round} 轮  |  "
                        f"剩余时间: {(deadline - time.time())/60:.1f}min")

            candidates = collect_candidate_urls(page, label=f"从容#{calm_round}")
            fresh = [c for c in candidates if c["url"] not in replied_urls]
            logger.info(f"新鲜候选: {len(fresh)} 个")

            if not fresh:
                wait = min(CONFIG["calm_rescan_interval"],
                           max(0, int(deadline - time.time())))
                if wait <= 0:
                    break
                logger.info(f"全是处理过的帖子，等 {wait}s 再刷...")
                time.sleep(wait)
                calm_round += 1
                continue

            new = run_scan_loop(
                page, fresh, replied_urls, replied_count,
                is_panic_fn=is_panic_fn,
                deadline=deadline,
                vector_store=vector_store, emb_model=emb_model, llm=llm,
                low_score_urls=low_score_urls,
            )
            replied_count += new

            if replied_count >= CONFIG["max_reply_actions"]:
                break

            if new == 0:
                wait = min(CONFIG["calm_rescan_interval"],
                           max(0, int(deadline - time.time())))
                if wait <= 0:
                    break
                logger.info(f"本轮没找到合适帖子，等 {wait}s 后回刷...")
                time.sleep(wait)

            if is_panic_fn():
                break

            calm_round += 1

        # ── 急行军扫描 ─────────────────────────────────────
        if time.time() < deadline and replied_count < CONFIG["max_reply_actions"]:
            logger.info(f"\n{'═'*50}")
            logger.info(f"🚨 急行军！剩余时间: {(deadline - time.time())/60:.1f}min")
            logger.info(f"   （累计回复 ≥ 总预算 × {CONFIG['panic_ratio']*100:.0f}%）")
            logger.info(f"{'═'*50}")

            panic_candidates = collect_candidate_urls(page, label="急行军")
            fresh = [c for c in panic_candidates if c["url"] not in replied_urls]
            run_scan_loop(
                page, fresh, replied_urls, replied_count,
                is_panic_fn=lambda: True,   # 急行军阶段始终是急行军
                deadline=deadline,
                vector_store=vector_store, emb_model=emb_model, llm=llm,
            )

    elapsed = (time.time() - phase_start) / 60
    logger.info(f"💬 回复阶段结束，耗时 {elapsed:.1f} 分钟，本轮发送 {replied_count} 次")


# ══════════════════════════════════════════════
#  单独运行入口
# ══════════════════════════════════════════════
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    logger.info("🧠 加载 Embedding 模型...")
    _emb_model    = SentenceTransformer(CONFIG["embedding_model"])
    _vector_store = InMemoryVectorStore(CONFIG["db_name"], _emb_model)
    _llm          = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])
    logger.info("✅ 就绪\n")

    # 单独运行时：一次性跑 reply_minutes 分钟，不限累计时间（全程从容）
    _deadline = time.time() + CONFIG["reply_minutes"] * 60
    auto_crawler(
        deadline=_deadline,
        cum_reply_secs_at_start=0.0,
        total_reply_budget=float("inf"),  # 不限制，全程从容
        vector_store=_vector_store,
        emb_model=_emb_model,
        llm=_llm,
    )