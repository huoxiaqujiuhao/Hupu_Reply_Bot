"""
basketball_scraper.py — 篮球区历史语料采集（建库用）

采集来源：
  - bbs.hupu.com/502  （篮球资讯，按最新回复时间排序）
  - bbs.hupu.com/vote （讨论/投票贴）

目的：建立篮球区帖子语料库，用于：
  1. 统计高频球星名，确定 basketball_player_list
  2. 作为 RAG 参考库的冷启动语料

运行：python basketball_scraper.py
采集完毕后运行 basketball_analyze.py 查看高频球星统计。
"""
from playwright.sync_api import sync_playwright
import sqlite3
import json
import time
import random
import re
import html as html_module
import os
from config import CONFIG, get_logger

logger = get_logger("BballScraper")

MIN_REPLIES = 100  # 篮球区入库门槛（比步行街宽松，但确保有一定讨论量）
SECTION     = "basketball"

# 两个来源：(第1页URL, 第N页URL模板)
SOURCES = [
    {
        "name":      "篮球资讯",
        "page1":     "https://bbs.hupu.com/502",
        "page_n":    "https://bbs.hupu.com/502-{page}",
        "progress_key": "bball_502_page",
    },
    {
        "name":      "投票讨论",
        "page1":     "https://bbs.hupu.com/vote",
        "page_n":    "https://bbs.hupu.com/vote-{page}",
        "progress_key": "bball_vote_page",
    },
]


# ══════════════════════════════════════════════
#  数据库初始化 + migration
# ══════════════════════════════════════════════
def init_db() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(CONFIG["db_name"])
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # 确保 Posts 表有 section 列
    for ddl in [
        "ALTER TABLE Posts ADD COLUMN section TEXT DEFAULT 'stepstreet'",
    ]:
        try:
            conn.execute(ddl)
            conn.commit()
            logger.info("✅ Posts.section 列已添加")
        except Exception:
            pass  # 列已存在

    # 进度表（和主爬虫共用同一张表，但 key 不同）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS CrawlProgress (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def load_progress(conn, key: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT value FROM CrawlProgress WHERE key=?", (key,))
    row = cur.fetchone()
    return int(row[0]) if row else 1


def save_progress(conn, key: str, page_num: int):
    conn.execute(
        "INSERT OR REPLACE INTO CrawlProgress (key, value) VALUES (?, ?)",
        (key, str(page_num))
    )
    conn.commit()


# ══════════════════════════════════════════════
#  工具
# ══════════════════════════════════════════════
def clean_html(raw: str) -> str:
    if not raw: return ""
    raw = re.sub(r'<img[^>]*>', '[图片]', raw)
    raw = re.sub(r'<[^>]+>', '', raw)
    raw = html_module.unescape(raw)
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


def human_like_scroll(page):
    try:
        y = random.randint(*CONFIG["scroll_y_range"])
        page.evaluate(f"window.scrollTo(0, {y})")
        time.sleep(random.uniform(*CONFIG["scroll_pause_1"]))
        page.evaluate(f"window.scrollTo(0, {y + random.randint(100, 400)})")
        time.sleep(random.uniform(*CONFIG["scroll_pause_2"]))
    except Exception:
        pass


def random_sleep(range_tuple: tuple, label: str = ""):
    t = random.uniform(*range_tuple)
    logger.info(f"💤 {label} {t:.1f}s")
    time.sleep(t)


# ══════════════════════════════════════════════
#  帖子解析（不过滤年龄，建库阶段要历史数据）
# ══════════════════════════════════════════════
def parse_post(page, url: str) -> dict | None:
    data = get_next_data(page)
    if not data:
        return None
    try:
        detail         = data["props"]["pageProps"]["detail"]
        thread         = detail["thread"]
        post_time_str  = thread.get("createdAtFormat", "")
        post_timestamp = int(thread.get("createdAt", 0)) // 1000
        raw_content    = thread.get("content", "")
        reply_count    = detail["replies"]["count"]

        if reply_count < MIN_REPLIES:
            logger.info(f"回复不足 {MIN_REPLIES}（{reply_count}），跳过")
            return None

        has_image = "<img" in raw_content

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
#  单个来源爬取
# ══════════════════════════════════════════════
def crawl_source(source: dict, conn: sqlite3.Connection, page, max_posts: int, deadline: float | None):
    cur         = conn.cursor()
    progress_key = source["progress_key"]
    page_num    = load_progress(conn, progress_key)
    harvested   = 0
    consecutive_errors = 0

    logger.info(f"\n{'='*55}")
    logger.info(f"📋 开始采集来源：{source['name']}，从第 {page_num} 页")
    logger.info(f"{'='*55}")

    while harvested < max_posts:
        if deadline and time.time() >= deadline:
            logger.info("⏰ 时间到，停止")
            break

        list_url = (
            source["page1"] if page_num == 1
            else source["page_n"].format(page=page_num)
        )
        logger.info(f"📋 [{source['name']}] 第 {page_num} 页: {list_url}")

        # 列表页加载（带重试）
        page_loaded = False
        for attempt in range(1, CONFIG["max_page_retries"] + 1):
            if deadline and time.time() >= deadline:
                break
            try:
                page.goto(list_url, timeout=20000)
                human_like_scroll(page)
                page_loaded = True
                consecutive_errors = 0
                break
            except Exception as e:
                logger.warning(f"列表页加载失败（第{attempt}次）: {e}")
                if attempt < CONFIG["max_page_retries"]:
                    random_sleep(CONFIG["scraper_delay_on_error"], "错误冷却")

        if not page_loaded:
            consecutive_errors += 1
            if consecutive_errors >= CONFIG["max_consecutive_errors"]:
                logger.warning("连续错误过多，触发长休眠")
                random_sleep(CONFIG["long_sleep_on_errors"], "长休眠")
                consecutive_errors = 0
            page_num += 1
            save_progress(conn, progress_key, page_num)
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

        if not candidates:
            logger.info("本页无候选，可能已到末页，停止此来源")
            break

        for url in candidates:
            if deadline and time.time() >= deadline:
                break
            if harvested >= max_posts:
                break

            cur.execute("SELECT 1 FROM Posts WHERE url=?", (url,))
            if cur.fetchone():
                continue

            logger.info(f"[{harvested}] 进帖: {url}")
            try:
                page.goto(url, timeout=20000)
                human_like_scroll(page)
                post_data = parse_post(page, url)

                if post_data:
                    cur.execute('''
                        INSERT OR IGNORE INTO Posts
                            (url, title, content, post_time_str, post_timestamp,
                             total_replies, has_image, section)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        post_data["url"], post_data["title"], post_data["content"],
                        post_data["post_time_str"], post_data["post_timestamp"],
                        post_data["total_replies"], post_data["has_image"],
                        SECTION,
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
                        f"✅ 收录｜{post_data['title'][:30]}  "
                        f"回复:{post_data['total_replies']}  亮评:{len(post_data['top_lights'])}条"
                    )

            except Exception as e:
                logger.error(f"抓取失败: {e}", exc_info=True)
                consecutive_errors += 1
                if consecutive_errors >= CONFIG["max_consecutive_errors"]:
                    logger.warning("连续错误，长休眠")
                    random_sleep(CONFIG["long_sleep_on_errors"], "长休眠")
                    consecutive_errors = 0
                else:
                    random_sleep(CONFIG["scraper_delay_on_error"], "错误冷却")
                continue

            random_sleep(CONFIG["scraper_delay_posts"], "帖子间隔")

        page_num += 1
        save_progress(conn, progress_key, page_num)

        if deadline and time.time() >= deadline:
            break

        random_sleep(CONFIG["scraper_delay_pages"], "翻页间隔")

    logger.info(f"📦 [{source['name']}] 本次新增 {harvested} 条")
    return harvested


# ══════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════
def run(max_posts_per_source: int = 500, deadline: float = None):
    """
    max_posts_per_source: 每个来源最多采集多少条（两个来源合计最多 2x）
    deadline: 绝对时间戳，到点停止（给 main.py 调用时用）
    """
    logger.info("🏀 篮球区爬虫启动")
    conn  = init_db()
    total = 0

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page    = context.pages[0]

        for source in SOURCES:
            if deadline and time.time() >= deadline:
                break
            n = crawl_source(source, conn, page, max_posts_per_source, deadline)
            total += n

    conn.close()
    logger.info(f"🏀 篮球区爬虫结束，共新增 {total} 条帖子")


if __name__ == "__main__":
    # 单独运行：每个来源最多采集500条，合计约1000条用于分析
    run(max_posts_per_source=500)
