"""
test_scraper.py — 爬虫（数据采集端）
可被 main.py 调用，也可单独运行。
单独运行时按帖子数量上限控制；被 main.py 调用时按 deadline 控制。
"""
from playwright.sync_api import sync_playwright
import sqlite3
import json
import time
import random
import re
import html
import os
from datetime import datetime, timezone
from config import CONFIG, get_logger

logger = get_logger("Scraper")


# ══════════════════════════════════════════════
#  数据库引擎
# ══════════════════════════════════════════════
def init_database() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(CONFIG["db_name"])
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS Posts (
        url             TEXT PRIMARY KEY,
        title           TEXT NOT NULL,
        content         TEXT,
        post_time_str   TEXT,
        post_timestamp  INTEGER,
        total_replies   INTEGER,
        has_image       BOOLEAN,
        scraped_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ai_tag          TEXT,
        ai_reasoning    TEXT,
        cluster_id      INTEGER
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS Comments (
        pid                  TEXT PRIMARY KEY,
        post_url             TEXT,
        username             TEXT,
        content              TEXT,
        lights               INTEGER,
        comment_reply_count  INTEGER,
        comment_timestamp    INTEGER,
        time_diff_seconds    INTEGER,
        FOREIGN KEY(post_url) REFERENCES Posts(url)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS CrawlProgress (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    ''')
    conn.commit()
    return conn


# ══════════════════════════════════════════════
#  进度持久化（断点续爬）
# ══════════════════════════════════════════════
def load_progress(conn) -> int:
    cur = conn.cursor()
    cur.execute("SELECT value FROM CrawlProgress WHERE key='last_page'")
    row = cur.fetchone()
    return int(row[0]) if row else 1

def save_progress(conn, page_num: int):
    conn.execute(
        "INSERT OR REPLACE INTO CrawlProgress (key, value) VALUES ('last_page', ?)",
        (str(page_num),)
    )
    conn.commit()

def reset_progress(conn):
    conn.execute("DELETE FROM CrawlProgress WHERE key='last_page'")
    conn.commit()


# ══════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════
def clean_html(raw: str) -> str:
    if not raw: return ""
    raw = re.sub(r'<img[^>]*>', '[图片]', raw)
    raw = re.sub(r'<[^>]+>', '', raw)
    raw = html.unescape(raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    return raw.strip()

def get_next_data(page) -> dict | None:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=15000)
        page.wait_for_selector("#__NEXT_DATA__", state="attached", timeout=15000)
        return json.loads(page.locator("#__NEXT_DATA__").inner_text())
    except Exception as e:
        logger.warning(f"__NEXT_DATA__ 读取失败: {str(e)[:80]}")
        return None

def human_like_behavior(page):
    try:
        y = random.randint(300, 1200)
        page.evaluate(f"window.scrollTo(0, {y})")
        time.sleep(random.uniform(0.5, 1.5))
        page.evaluate(f"window.scrollTo(0, {y + random.randint(100, 400)})")
        time.sleep(random.uniform(0.3, 0.8))
    except Exception:
        pass

def random_sleep(range_tuple: tuple, label: str = "休眠"):
    t = random.uniform(*range_tuple)
    logger.info(f"💤 {label} {t:.1f}s")
    time.sleep(t)

def is_post_fresh(post_timestamp: int) -> bool:
    if not post_timestamp: return False
    now_ts      = int(datetime.now(timezone.utc).timestamp())
    age_seconds = now_ts - post_timestamp
    return 0 <= age_seconds <= CONFIG["post_max_age_hours"] * 3600

def try_recover_page(page):
    logger.warning("尝试恢复页面...")
    try:
        page.goto("https://bbs.hupu.com/topic-daily", timeout=15000)
        page.wait_for_load_state("domcontentloaded", timeout=10000)
        logger.info("页面已恢复")
        return True
    except Exception as e:
        logger.error(f"页面恢复失败: {e}")
        return False


# ══════════════════════════════════════════════
#  帖子解析
# ══════════════════════════════════════════════
def parse_post(page, url: str) -> dict | None:
    data = get_next_data(page)
    if not data: return None
    try:
        detail         = data["props"]["pageProps"]["detail"]
        thread         = detail["thread"]
        post_time_str  = thread.get("createdAtFormat", "")
        post_timestamp = int(thread.get("createdAt", 0)) // 1000
        raw_content    = thread.get("content", "")
        reply_count    = detail["replies"]["count"]

        if not is_post_fresh(post_timestamp):
            logger.info(f"帖子超龄（{post_time_str}），跳过")
            return None
        if reply_count < CONFIG["harvest_min_replies"]:
            logger.info(f"回复太少（{reply_count}），跳过")
            return None

        has_image = "<img" in raw_content
        if not CONFIG["allow_image_posts"] and has_image:
            return None

        top_lights = []
        for light in detail.get("lights", [])[:CONFIG["scraper_top_lights"]]:
            c = clean_html(light.get("content", ""))
            if not c or c.strip() == "[图片]":
                continue
            comment_ts = int(light.get("createdAt", 0)) // 1000
            top_lights.append({
                "pid":                 light.get("pid", ""),
                "username":            light["author"]["puname"],
                "content":             c,
                "lights":              light.get("allLightCount", 0),
                "comment_reply_count": int(light.get("replyNum", 0)),
                "comment_timestamp":   comment_ts,
                "time_diff_seconds":   max(0, comment_ts - post_timestamp),
            })

        return {
            "url":            url,
            "title":          thread["title"],
            "content":        clean_html(raw_content),
            "post_time_str":  post_time_str,
            "post_timestamp": post_timestamp,
            "total_replies":  reply_count,
            "has_image":      has_image,
            "top_lights":     top_lights,
        }
    except Exception as e:
        logger.error(f"解析异常: {e}", exc_info=True)
        return None


# ══════════════════════════════════════════════
#  主控台
#  deadline: 绝对时间戳，到点停止
#           None = 只按 max_harvest_posts 控制（单独运行时）
# ══════════════════════════════════════════════
def auto_crawler(deadline: float = None, max_posts: int = None,
                 bball_drain_fn=None, replied_urls: set = None):
    if max_posts is None:
        max_posts = 999999  # 被 main.py 调用时不按数量限制，只按时间
    if replied_urls is None:
        replied_urls = set()

    logger.info("🤖 爬虫启动，连接浏览器...")
    conn = init_database()
    cur  = conn.cursor()

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page    = context.pages[0] if context.pages else context.new_page()

        cur.execute("SELECT COUNT(*) FROM Posts")
        harvested = cur.fetchone()[0]
        logger.info(f"📊 数据库已有 {harvested} 个帖子")

        page_num = 1

        consecutive_errors = 0

        while harvested < max_posts:
            # ⏰ 时间检查
            if deadline and time.time() >= deadline:
                logger.info("⏰ 爬虫时间到，本轮结束")
                break

            list_url = (
                "https://bbs.hupu.com/topic-daily"
                if page_num == 1
                else f"https://bbs.hupu.com/topic-daily-{page_num}"
            )
            logger.info(f"📋 列表第 {page_num} 页: {list_url}")

            # 列表页加载（带重试）
            page_loaded = False
            for attempt in range(1, CONFIG["max_page_retries"] + 1):
                if deadline and time.time() >= deadline:
                    break
                try:
                    page.goto(list_url, timeout=20000)
                    human_like_behavior(page)
                    page_loaded = True
                    consecutive_errors = 0
                    break
                except Exception as e:
                    logger.warning(f"列表页加载失败（第{attempt}次）: {e}")
                    if "closed" in str(e).lower():
                        raise RuntimeError("Chrome 已关闭，终止运行") from e
                    if attempt < CONFIG["max_page_retries"]:
                        random_sleep(CONFIG["scraper_delay_on_error"], "错误冷却")
                        try_recover_page(page)

            if not page_loaded:
                logger.error(f"第 {page_num} 页连续失败，跳过")
                consecutive_errors += 1
                if consecutive_errors >= CONFIG["max_consecutive_errors"]:
                    logger.warning("连续错误过多，触发长休眠")
                    random_sleep(CONFIG["long_sleep_on_errors"], "长休眠")
                    consecutive_errors = 0
                page_num += 1
                save_progress(conn, page_num)
                continue

            # 提取候选帖子链接
            candidates = []
            seen: set  = set()
            for el in page.locator("a[href]").all():
                href = el.get_attribute("href")
                if href and re.match(r'^/\d+\.html$', href):
                    full_url = "https://bbs.hupu.com" + href
                    if full_url not in seen:
                        seen.add(full_url)
                        candidates.append(full_url)
            logger.info(f"🎯 本页候选: {len(candidates)} 个")

            for url in candidates:
                if deadline and time.time() >= deadline:
                    logger.info("⏰ 时间到，停止本页剩余")
                    break
                if harvested >= max_posts:
                    break

                cur.execute("SELECT 1 FROM Posts WHERE url=?", (url,))
                if cur.fetchone():
                    continue

                logger.info(f"[{harvested}] 进帖: {url}")
                try:
                    page.goto(url, timeout=20000)
                    human_like_behavior(page)
                    post_data = parse_post(page, url)

                    if post_data:
                        cur.execute('''
                            INSERT INTO Posts
                                (url, title, content, post_time_str, post_timestamp, total_replies, has_image)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            post_data["url"], post_data["title"], post_data["content"],
                            post_data["post_time_str"], post_data["post_timestamp"],
                            post_data["total_replies"], post_data["has_image"],
                        ))
                        for c in post_data["top_lights"]:
                            cur.execute('''
                                INSERT OR IGNORE INTO Comments
                                    (pid, post_url, username, content, lights,
                                     comment_reply_count, comment_timestamp, time_diff_seconds)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                c["pid"], post_data["url"], c["username"], c["content"],
                                c["lights"], c["comment_reply_count"],
                                c["comment_timestamp"], c["time_diff_seconds"],
                            ))
                        conn.commit()
                        harvested += 1
                        consecutive_errors = 0
                        logger.info(
                            f"✅ 收录｜{post_data['title'][:25]}  "
                            f"回复:{post_data['total_replies']}  亮评:{len(post_data['top_lights'])}条"
                        )

                except Exception as e:
                    logger.error(f"抓取失败: {e}", exc_info=True)
                    consecutive_errors += 1
                    if consecutive_errors >= CONFIG["max_consecutive_errors"]:
                        logger.warning("连续错误，长休眠并恢复页面")
                        random_sleep(CONFIG["long_sleep_on_errors"], "长休眠")
                        try_recover_page(page)
                        consecutive_errors = 0
                    else:
                        random_sleep(CONFIG["scraper_delay_on_error"], "错误冷却")
                    continue

                # 篮球绝对优先：每收录一帖后立即消费篮球队列
                if bball_drain_fn:
                    bball_count = bball_drain_fn(page, replied_urls)
                else:
                    bball_count = 0

                if bball_count == 0:
                    random_sleep(CONFIG["scraper_delay_posts"], "帖子间隔")

            if deadline and time.time() >= deadline:
                break

            page_num += 1
            save_progress(conn, page_num)
            random_sleep(CONFIG["scraper_delay_pages"], "翻页间隔")

        logger.info(f"🕷️ 爬虫结束，本次新增后数据库共 {harvested} 个帖子")
        conn.close()


if __name__ == "__main__":
    # 单独运行：不按时间，按帖子数量上限（可在 config.py 里改 harvest_min_replies 等参数）
    auto_crawler(max_posts=1000)