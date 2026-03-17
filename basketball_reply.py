# -*- coding: utf-8 -*-
"""
basketball_reply.py -- basketball section real-time reply pipeline

BballScanner: background thread, anonymous HTTP scan every 30s,
              scores posts and fills a priority queue
drain_bball_queue: called by main thread at stepstreet breakpoints
bball_rag_generate: rag_generate + slang injection
"""
import re
import time
import json
import sqlite3
import threading
import requests
import numpy as np
from queue import PriorityQueue, Empty
from openai import OpenAI
from config import CONFIG, get_logger
from embedder import EmbeddingModel
import memory_store
import reply_bot

logger = get_logger("BballReply")

BBALL_SOURCES = [
    "https://bbs.hupu.com/502-postdate",
    "https://bbs.hupu.com/vote-postdate",
]

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# ══════════════════════════════════════════════
#  Global state
# ══════════════════════════════════════════════
_bball_queue:     PriorityQueue  = PriorityQueue()   # (-score, ts, item_dict)
_seen_bball_urls: set            = set()              # never re-queue same URL
_subject_cooldown: dict          = {}                 # {subject: last_replied_ts}
_state_lock:      threading.Lock = threading.Lock()


# ══════════════════════════════════════════════
#  Tier map (built from DB at startup)
# ══════════════════════════════════════════════
_tier_map: dict[str, float] = {}


def build_tier_map():
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("""
        SELECT SUBSTR(ai_tag, 1, INSTR(ai_tag, '-') - 1) as subject,
               COUNT(*) as n
        FROM Posts
        WHERE section='basketball' AND ai_tag != 'other' AND ai_tag LIKE '%-%'
        GROUP BY subject
    """)
    rows = cur.fetchall()
    conn.close()

    t1 = CONFIG["basketball_tier1_min_posts"]
    t2 = CONFIG["basketball_tier2_min_posts"]
    new_map = {}
    for subject, n in rows:
        if n >= t1:
            new_map[subject] = CONFIG["basketball_tier1_multiplier"]
        elif n >= t2:
            new_map[subject] = CONFIG["basketball_tier2_multiplier"]
        else:
            new_map[subject] = CONFIG["basketball_tier3_multiplier"]

    _tier_map.update(new_map)
    t1c = sum(1 for v in new_map.values() if v >= CONFIG["basketball_tier1_multiplier"])
    t2c = sum(1 for v in new_map.values() if CONFIG["basketball_tier2_multiplier"] <= v < CONFIG["basketball_tier1_multiplier"])
    logger.info(
        "Tier map built: T1={} | T2={} | T3={}".format(
            t1c, t2c, len(new_map) - t1c - t2c
        )
    )


def _get_tier(subject) -> float:
    return _tier_map.get(subject, CONFIG["basketball_tier3_multiplier"])


# ══════════════════════════════════════════════
#  Subject extraction from title
# ══════════════════════════════════════════════
def _extract_subject(title: str, url: str):
    # DB first (post already classified)
    try:
        conn = sqlite3.connect(CONFIG["db_name"])
        cur  = conn.cursor()
        cur.execute("SELECT ai_tag FROM Posts WHERE url=? AND section='basketball'", (url,))
        row = cur.fetchone()
        conn.close()
        if row and row[0] and '-' in row[0] and row[0] != 'other':
            return row[0].rsplit('-', 1)[0]
    except Exception:
        pass

    # Keyword match (longest subject name first to avoid partial hits)
    for subject in sorted(_tier_map.keys(), key=len, reverse=True):
        if subject in title:
            return subject
    return None


# ══════════════════════════════════════════════
#  Anonymous HTTP list fetch
# ══════════════════════════════════════════════
def _fetch_list(source_url: str) -> list[dict]:
    """
    Supports two page formats:
    - Next.js (__NEXT_DATA__ JSON): bbs.hupu.com/502, /vote
    - Old HTML (bbs-sl-web-post-body li): bbs.hupu.com/502-postdate, /vote-postdate
    """
    try:
        resp = requests.get(source_url, headers=_HTTP_HEADERS, timeout=10)
        resp.raise_for_status()

        # Try Next.js format first
        m = re.search(
            r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
            resp.text, re.DOTALL
        )
        if m:
            data    = json.loads(m.group(1))
            threads = (data.get("props", {})
                           .get("pageProps", {})
                           .get("forumData", {})
                           .get("threads", []))
            results = []
            for t in threads:
                tid = t.get("tid") or t.get("id")
                if not tid:
                    continue
                url       = "https://bbs.hupu.com/{}.html".format(tid)
                title     = t.get("subject") or t.get("title") or ""
                post_time = t.get("createdAtFormat") or t.get("createTime") or ""
                results.append({"url": url, "title": title, "post_time": post_time})
            return results

        # Old HTML format: parse <li class="bbs-sl-web-post-body"> blocks
        results = []
        blocks = re.findall(
            r'<li class="bbs-sl-web-post-body">(.*?)</li>',
            resp.text, re.DOTALL
        )
        for block in blocks:
            href_m  = re.search(r'href="(/\d+\.html)"', block)
            title_m = re.search(r'class="p-title"[^>]*>([^<]+)<', block)
            time_m  = re.search(r'class="post-time">([^<]+)<', block)
            if not href_m or not title_m:
                continue
            url   = "https://bbs.hupu.com" + href_m.group(1)
            title = title_m.group(1).strip()
            post_time = time_m.group(1).strip() if time_m else ""
            results.append({"url": url, "title": title, "post_time": post_time})
        return results

    except Exception as e:
        logger.warning("List fetch failed {}: {}".format(source_url, e))
        return []


def _parse_minutes_ago(time_str: str) -> float:
    """Parse hupu time string to minutes ago. Handles both relative and absolute formats."""
    import datetime
    if not time_str:
        return 999.0
    # Relative: 刚刚 / X秒前
    if "\u521a\u521a" in time_str or "\u79d2" in time_str:
        return 0.5
    # Relative: X分钟前
    m = re.search(r'(\d+)\s*\u5206\u949f', time_str)
    if m:
        return float(m.group(1))
    # Relative: X小时前
    m = re.search(r'(\d+)\s*\u5c0f\u65f6', time_str)
    if m:
        return float(m.group(1)) * 60
    # Absolute: MM-DD HH:MM  (e.g. "03-17 05:43")
    # Hupu server is UTC+8; compare against UTC+8 "now" to avoid timezone gaps
    m = re.match(r'(\d{2})-(\d{2})\s+(\d{2}):(\d{2})', time_str)
    if m:
        import datetime as _dt
        utc8_now = _dt.datetime.utcnow() + _dt.timedelta(hours=8)
        month, day, hour, minute = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        post_dt = utc8_now.replace(month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
        if post_dt > utc8_now + _dt.timedelta(minutes=5):  # clearly future → last year
            post_dt = post_dt.replace(year=utc8_now.year - 1)
        delta = (utc8_now - post_dt).total_seconds() / 60.0
        return max(0.0, delta)
    return 999.0


# ══════════════════════════════════════════════
#  Priority scoring
# ══════════════════════════════════════════════
def _score(subject, minutes_ago: float) -> float:
    tier = _get_tier(subject) if subject else CONFIG["basketball_tier3_multiplier"]
    return tier / (minutes_ago + 1.0)


# ══════════════════════════════════════════════
#  Background scanner thread
# ══════════════════════════════════════════════
class BballScanner(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="BballScanner")
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        logger.info("Basketball background scanner started")
        while not self._stop.is_set():
            try:
                self._scan()
            except Exception as e:
                logger.error("Scanner error: {}".format(e))
            self._stop.wait(CONFIG["basketball_scan_interval"])

    def _scan(self):
        candidates = []
        for src in BBALL_SOURCES:
            candidates.extend(_fetch_list(src))

        if not candidates:
            return

        with _state_lock:
            new_posts = [p for p in candidates if p["url"] not in _seen_bball_urls]

        if not new_posts:
            return

        # Score all new posts
        scored = []
        for p in new_posts:
            minutes  = _parse_minutes_ago(p["post_time"])
            subject  = _extract_subject(p["title"], p["url"])
            score    = _score(subject, minutes)
            scored.append((score, p, subject))

        scored.sort(key=lambda x: -x[0])

        max_per_scan    = CONFIG["basketball_max_queue_per_scan"]
        max_per_subject = CONFIG["basketball_max_per_subject_per_scan"]
        added           = 0
        subject_count: dict = {}

        with _state_lock:
            for score, post, subject in scored:
                if added >= max_per_scan:
                    break
                url = post["url"]
                if url in _seen_bball_urls:
                    continue

                subj_key = subject or "__unknown__"
                if subject_count.get(subj_key, 0) >= max_per_subject:
                    continue

                # Same-subject cooldown
                if subject and subject in _subject_cooldown:
                    elapsed = time.time() - _subject_cooldown[subject]
                    if elapsed < CONFIG["basketball_subject_cooldown"]:
                        continue

                _seen_bball_urls.add(url)
                subject_count[subj_key] = subject_count.get(subj_key, 0) + 1
                item = {
                    "url":       url,
                    "title":     post["title"],
                    "subject":   subject,
                    "score":     score,
                    "queued_at": time.time(),
                }
                _bball_queue.put((-score, time.time(), item))
                added += 1
                logger.info(
                    "  [BBALL] queued [{}] {} score={:.2f}".format(
                        subject or "?", post["title"][:30], score
                    )
                )

        if added > 0:
            logger.info("Basketball scan: {} new posts queued".format(added))


def get_queue_size() -> int:
    return _bball_queue.qsize()


def mark_subject_replied(subject):
    if subject:
        with _state_lock:
            _subject_cooldown[subject] = time.time()


# ══════════════════════════════════════════════
#  Slang injection
# ══════════════════════════════════════════════
def _build_slang_block(subject: str, query_text: str, emb_model: EmbeddingModel) -> str:
    if not subject:
        return ""
    top_k = CONFIG.get("basketball_slang_top_k", 8)
    terms = memory_store.search_slang(subject, query_text, emb_model, top_k=top_k)
    if not terms:
        return ""

    by_stance: dict = {}
    for t in terms:
        by_stance.setdefault(t["stance"], []).append(t["term"])

    lines  = ["[{}circle slang (pick naturally, not all required)]".format(subject)]
    labels = {"fan": "fan", "hater": "hater", "neutral": "neutral"}
    # Build Chinese label string without hardcoding Chinese in source
    fan_label     = "\u652f\u6301\u8005\u7528\u8bed"   # 支持者用语
    hater_label   = "\u9ed1\u5b50\u7528\u8bed"         # 黑子用语
    neutral_label = "\u4e2d\u7acb\u7528\u8bed"         # 中立用语
    stance_cfg = [("fan", fan_label), ("hater", hater_label), ("neutral", neutral_label)]

    header = "[{}\u5708\u5b50\u9ed1\u8bdd\uff08\u6309\u60c5\u51b5\u81ea\u7136\u9009\u7528\uff0c\u4e0d\u5fc5\u5168\u7528\uff09]".format(subject)
    lines  = [header]
    for stance, label in stance_cfg:
        if stance in by_stance:
            lines.append("  {}: {}".format(label, "\u3001".join(by_stance[stance][:5])))
    return "\n".join(lines)


# ══════════════════════════════════════════════
#  Basketball RAG generate
# ══════════════════════════════════════════════
def bball_rag_generate(
    title: str,
    content: str,
    ai_tag: str,
    subject: str,
    emb_model: EmbeddingModel,
    llm: OpenAI,
    current_replies: list = None,
    vector_store=None,
) -> str:
    query_text = "{}.{}".format(title, (content or "")[:CONFIG["post_text_max_chars"]])

    # Reference block from similar historical posts
    ref_block  = "  (no similar historical comments)"
    strong_ref = False
    if vector_store is not None:
        dense_mat, sp      = emb_model.encode_hybrid([query_text])
        q_dense, q_sparse  = dense_mat[0], sp[0]
        similar   = [p for p in vector_store.query(q_dense, q_sparse, ai_tag, CONFIG["top_k_posts"])
                     if p["similarity"] >= CONFIG["memory_case_sim_threshold"]]
        comments  = reply_bot.fetch_top_comments([p["url"] for p in similar])
        if len(similar) >= 2:
            strong_ref = (similar[0]["similarity"] - similar[1]["similarity"]) >= 0.10
        elif len(similar) == 1:
            strong_ref = similar[0]["similarity"] >= 0.88
        if comments:
            ref_block = "\n".join([
                "  {}. [{}]({} likes)".format(
                    i + 1,
                    c["comment"][:CONFIG["comment_ref_max_chars"]],
                    c["lights"]
                )
                for i, c in enumerate(comments)
            ])

    # Positive cases
    query_for_cases = "{}.{}".format(title, (content or "")[:200])
    sim_th = CONFIG["memory_case_sim_threshold"]
    pos_cases = memory_store.search_cases(
        query_text=query_for_cases, case_type="positive",
        emb_model=emb_model, category=ai_tag,
        top_k=CONFIG["memory_top_k_cases"],
    )
    pos_cases = [c for c in pos_cases if c["similarity"] >= sim_th]

    # Negative cases
    neg_cases = []
    for ct in ("negative_content", "negative_duplicate"):
        hits = memory_store.search_cases(
            query_text=query_for_cases, case_type=ct,
            emb_model=emb_model, category=ai_tag,
            top_k=CONFIG["memory_top_k_neg"],
        )
        neg_cases += [c for c in hits if c["similarity"] >= sim_th]
    seen_r, deduped_neg = set(), []
    for c in sorted(neg_cases, key=lambda x: -x["similarity"]):
        k = c["bot_reply"][:60]
        if k not in seen_r:
            seen_r.add(k)
            deduped_neg.append(c)
        if len(deduped_neg) >= CONFIG["memory_top_k_neg_inject"]:
            break

    # Current vibe
    current_vibe_block = (
        "\n".join(["  - {}".format(r) for r in current_replies])
        if current_replies else "  (no replies yet)"
    )

    # Slang block
    slang_block = _build_slang_block(subject, query_text, emb_model)

    # Negative warning
    neg_warn_block = "\n".join([
        "- [{}] ({}, {} likes)".format(
            c["bot_reply"][:80],
            "content direction lost" if c["case_type"] == "negative_content" else "duplicate of top comment",
            c["light_count"],
        )
        for c in deduped_neg
    ]) if deduped_neg else ""

    # Step 1: plan call
    plan = reply_bot._plan_call(
        title, content, ai_tag, ref_block, current_vibe_block, neg_warn_block, llm
    )
    logger.info("Plan: vibe={} | angle={}".format(
        plan.get("vibe", "")[:30], plan.get("angle", "")[:40]
    ))

    plan_block = ""
    if plan.get("angle") or plan.get("hook"):
        plan_block = (
            "{}\n".format(CONFIG["prompt_plan_block_header"])
            + "wind: {}\n".format(plan.get("vibe", ""))
            + "angle: {}\n".format(plan.get("angle", ""))
            + "hook: {}\n\n".format(plan.get("hook", ""))
        )

    # Step 2: generate call
    memory_block = ""
    if pos_cases:
        case_lines = [
            "- post[{}]\n  your comment: [{}] ({} likes)".format(
                c["post_title"][:30], c["bot_reply"][:100], c["light_count"]
            )
            for c in pos_cases
        ]
        memory_block = CONFIG["prompt_pos_case_header"] + "\n".join(case_lines)

    system = CONFIG["prompt_basketball_generate_system"] + memory_block
    user   = (
        "[category]{}\n[title]{}\n[content]{}\n\n".format(
            ai_tag, title, (content or "")[:CONFIG["post_prompt_max_chars"]]
        )
        + ("{}\n\n".format(slang_block) if slang_block else "")
        + plan_block
        + "{}\n{}\n\n".format(
            CONFIG["prompt_ref_strong_label"] if strong_ref else CONFIG["prompt_ref_weak_label"],
            ref_block,
        )
        + CONFIG["prompt_generate_suffix"]
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
#  Process single basketball post (main thread)
# ══════════════════════════════════════════════
def process_bball_post(
    page,
    item: dict,
    replied_urls: set,
    emb_model: EmbeddingModel,
    llm: OpenAI,
    vector_store=None,
) -> bool:
    url     = item["url"]
    subject = item["subject"]

    # Age cutoff (time in queue + post age)
    age_secs = time.time() - item["queued_at"]
    if age_secs > CONFIG["basketball_age_cutoff_minutes"] * 60:
        logger.info("[BBALL] stale ({:.1f}min in queue), skip: {}".format(
            age_secs / 60, item["title"][:30]
        ))
        return False

    if url in replied_urls:
        return False

    logger.info("\n[BBALL] processing [{}] {}".format(subject or "?", item["title"][:40]))

    try:
        page.goto(url, timeout=15000)
        reply_bot.human_like_behavior(page)
        post_data = reply_bot.parse_post(page)
        if not post_data:
            logger.warning("[BBALL] parse failed")
            replied_urls.add(url)
            return False

        # Confirm age after entering post (more accurate time)
        minutes_ago = reply_bot.parse_minutes_ago(post_data["post_time"])
        if minutes_ago > CONFIG["basketball_age_cutoff_minutes"]:
            logger.info("[BBALL] post too old ({:.0f}min), skip".format(minutes_ago))
            replied_urls.add(url)
            return False

        # Get ai_tag from DB or infer
        ai_tag = None
        try:
            conn = sqlite3.connect(CONFIG["db_name"])
            cur  = conn.cursor()
            cur.execute("SELECT ai_tag FROM Posts WHERE url=?", (url,))
            row = cur.fetchone()
            conn.close()
            if row and row[0] and '-' in row[0] and row[0] != 'other':
                ai_tag  = row[0]
                subject = ai_tag.rsplit('-', 1)[0]
        except Exception:
            pass

        if not ai_tag:
            ai_tag = "{}-neutral".format(subject) if subject else "basketball-neutral"

        logger.info("[BBALL] tag={} | {}min ago | {} replies".format(
            ai_tag, minutes_ago, post_data["total_replies"]
        ))

        # 在 LLM 调用前就写入 replied_urls，防止崩溃重启后重复发帖
        replied_urls.add(url)
        reply_bot.save_replied_history(replied_urls)

        reply_text = bball_rag_generate(
            title           = post_data["title"],
            content         = post_data["content"],
            ai_tag          = ai_tag,
            subject         = subject,
            emb_model       = emb_model,
            llm             = llm,
            current_replies = post_data["current_replies"],
            vector_store    = vector_store,
        )

        logger.info("[BBALL] reply: {}".format(reply_text))

        success = reply_bot.real_reply_action(page, reply_text)

        if success:
            mark_subject_replied(subject)
            try:
                tid = int(re.search(r'/(\d+)\.html', url).group(1))
                memory_store.save_reply_context(
                    tid               = tid,
                    content           = reply_text,
                    post_title        = post_data["title"],
                    post_content      = post_data.get("content", ""),
                    post_category     = ai_tag,
                    post_total_replies= post_data["total_replies"],
                )
            except Exception:
                pass

        return success

    except Exception as e:
        logger.error("[BBALL] error: {}".format(e), exc_info=True)
        replied_urls.add(url)
        return False


# ══════════════════════════════════════════════
#  Drain queue (called at stepstreet breakpoints)
# ══════════════════════════════════════════════
def drain_bball_queue(
    page,
    replied_urls: set,
    emb_model: EmbeddingModel,
    llm: OpenAI,
    vector_store=None,
) -> int:
    count = 0
    while True:
        try:
            _, _, item = _bball_queue.get_nowait()
        except Empty:
            break
        success = process_bball_post(page, item, replied_urls, emb_model, llm, vector_store)
        if success:
            count += 1
        if not _bball_queue.empty():
            reply_bot.random_sleep(CONFIG["reply_delay_posts"], "bball inter-post sleep")

    if count > 0:
        logger.info("[BBALL] drained queue, sent {} replies".format(count))
    return count
