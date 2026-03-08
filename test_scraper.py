from playwright.sync_api import sync_playwright
import sqlite3
import json
import time
import random
import re
import html
import logging
from datetime import datetime, timezone

# ══════════════════════════════════════════════
#  总控台（所有开关都在这里）
# ══════════════════════════════════════════════
CONFIG = {
    # 🎯 目标设定
    "max_harvest_posts": 1000,          # 彻夜开火目标：总共要收录多少个合格帖子？
    "db_name": "hupu_arsenal.db",       # SQLite 数据库的文件名

    # 🎯 筛选门槛
    "harvest_min_replies": 50,          # 帖子总回复数 >= 50 才收录
    "post_max_age_hours": 1024,           # 【修复】只收多少小时内的新帖（替代原脆弱字符串判断）
    "top_lights": 10,                   # 每个帖子最多收取前 N 条亮评
    "allow_image_posts": True,          # 【开关】是否收录带有图片的帖子？(True=全收，False=只要纯文字贴)

    # 🎯 节奏控制 (防封印休眠，彻夜运行建议稍微调大)
    "delay_between_posts": (4, 9),      # 进完一个帖子后的休息时间（秒）
    "delay_between_pages": (5, 10),     # 翻下一页列表时的休息时间（秒）
    "delay_on_error": (15, 30),         # 发生异常后的惩罚性休眠（秒）
    "max_consecutive_errors": 5,        # 连续错误超过此数，触发长休眠
    "long_sleep_on_errors": (60, 120),  # 连续出错后的长休眠（秒）
    "max_page_retries": 3,              # 单页加载失败后的最大重试次数
}

# ══════════════════════════════════════════════
#  日志系统（同时写终端 + 文件）
# ══════════════════════════════════════════════
def init_logger(db_name: str) -> logging.Logger:
    log_file = db_name.replace(".db", ".log")
    logger = logging.getLogger("HupuCrawler")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # 文件 handler（完整记录）
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # 终端 handler（只看 INFO 以上）
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ══════════════════════════════════════════════
#  数据库引擎 (自动建库建表 + WAL模式)
# ══════════════════════════════════════════════
def init_database(db_name: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_name)

    # 【修复】启用 WAL 模式，提升写入性能和崩溃安全性
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    cursor = conn.cursor()

    # 1. 帖子主表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Posts (
        url TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        content TEXT,
        post_time_str TEXT,
        post_timestamp INTEGER,
        total_replies INTEGER,
        has_image BOOLEAN,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ai_tag TEXT
    )
    ''')

    # 2. 亮评附表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Comments (
        pid TEXT PRIMARY KEY,
        post_url TEXT,
        username TEXT,
        content TEXT,
        lights INTEGER,
        comment_reply_count INTEGER,
        comment_timestamp INTEGER,
        time_diff_seconds INTEGER,
        FOREIGN KEY(post_url) REFERENCES Posts(url)
    )
    ''')

    # 3. 【新增】爬取进度记录表（断点续爬核心）
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS CrawlProgress (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    ''')

    conn.commit()
    return conn


# ══════════════════════════════════════════════
#  进度持久化（断点续爬）
# ══════════════════════════════════════════════
def load_progress(conn: sqlite3.Connection) -> int:
    """从数据库读取上次爬到第几页"""
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM CrawlProgress WHERE key='last_page'")
    row = cursor.fetchone()
    return int(row[0]) if row else 1

def save_progress(conn: sqlite3.Connection, page_num: int):
    """把当前页码写回数据库"""
    conn.execute(
        "INSERT OR REPLACE INTO CrawlProgress (key, value) VALUES ('last_page', ?)",
        (str(page_num),)
    )
    conn.commit()

def reset_progress(conn: sqlite3.Connection):
    """任务完成后清空进度，下次重新开始"""
    conn.execute("DELETE FROM CrawlProgress WHERE key='last_page'")
    conn.commit()


# ══════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════
def clean_html(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r'<img[^>]*>', '[图片]', raw)  # 保留图片语境标记
    raw = re.sub(r'<[^>]+>', '', raw)
    raw = html.unescape(raw)                     # 【修复】解码 &amp; &nbsp; &lt; 等 HTML 实体
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    return raw.strip()


def get_next_data(page, logger: logging.Logger) -> dict | None:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=15000)
        page.wait_for_selector("#__NEXT_DATA__", state="attached", timeout=15000)
        return json.loads(page.locator("#__NEXT_DATA__").inner_text())
    except Exception as e:
        logger.warning(f"数据读取超时或失败: {str(e)[:80]}")
        return None


def human_like_behavior(page):
    try:
        scroll_y = random.randint(300, 1200)
        page.evaluate(f"window.scrollTo(0, {scroll_y})")
        time.sleep(random.uniform(0.5, 1.5))
        page.evaluate(f"window.scrollTo(0, {scroll_y + random.randint(100, 400)})")
        time.sleep(random.uniform(0.3, 0.8))
    except Exception:
        pass  # 滚动报错不影响大局


def random_sleep(range_tuple: tuple, label: str = "隐蔽休眠", logger: logging.Logger = None):
    t = random.uniform(*range_tuple)
    msg = f"💤 {label} {t:.1f}s..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    time.sleep(t)


def is_post_fresh(post_timestamp: int) -> bool:
    """【修复】用时间戳数值判断帖子新鲜度，替代脆弱的字符串匹配"""
    if not post_timestamp:
        return False
    now_ts = int(datetime.now(timezone.utc).timestamp())
    age_seconds = now_ts - post_timestamp
    max_age_seconds = CONFIG["post_max_age_hours"] * 3600
    return 0 <= age_seconds <= max_age_seconds


def try_recover_page(page, logger: logging.Logger):
    """【新增】页面崩溃/卡死时的恢复尝试"""
    logger.warning("尝试恢复页面状态...")
    try:
        page.goto("https://bbs.hupu.com/topic-daily", timeout=15000)
        page.wait_for_load_state("domcontentloaded", timeout=10000)
        logger.info("页面已恢复到列表首页。")
        return True
    except Exception as e:
        logger.error(f"页面恢复失败: {e}")
        return False


# ══════════════════════════════════════════════
#  核心解析器
# ══════════════════════════════════════════════
def parse_post(page, top_n: int, url: str, logger: logging.Logger) -> dict | None:
    data = get_next_data(page, logger)
    if not data:
        return None

    try:
        detail = data["props"]["pageProps"]["detail"]
        thread = detail["thread"]

        post_time_str = thread.get("createdAtFormat", "")
        post_timestamp = int(thread.get("createdAt", 0)) // 1000
        raw_content = thread.get("content", "")
        reply_count = detail["replies"]["count"]

        # 1. 【修复】用时间戳数值判断新鲜度
        if not is_post_fresh(post_timestamp):
            logger.info(f"帖子已超过 {CONFIG['post_max_age_hours']}h（发帖时间: {post_time_str}），跳过。")
            return None

        # 2. 检查回复数是否达标
        if reply_count < CONFIG["harvest_min_replies"]:
            logger.info(f"太冷清（仅 {reply_count} 回复），跳过。")
            return None

        # 3. 图像贴开关控制
        has_image = "<img" in raw_content
        if not CONFIG["allow_image_posts"] and has_image:
            logger.info("包含图片（当前设置拒绝图像贴），跳过。")
            return None

        content_clean = clean_html(raw_content)

        top_lights = []
        for light in detail.get("lights", [])[:top_n]:
            comment_content = clean_html(light.get("content", ""))

            # 【修复】过滤空内容或纯图片亮评，对AI分析无价值
            if not comment_content or comment_content.strip() == "[图片]":
                continue

            comment_timestamp = int(light.get("createdAt", 0)) // 1000
            time_diff = max(0, comment_timestamp - post_timestamp)

            top_lights.append({
                "pid": light.get("pid", ""),
                "username": light["author"]["puname"],
                "content": comment_content,
                "lights": light.get("allLightCount", 0),
                "comment_reply_count": int(light.get("replyNum", 0)),
                "comment_timestamp": comment_timestamp,
                "time_diff_seconds": time_diff
            })

        return {
            "url": url,
            "title": thread["title"],
            "content": content_clean,
            "post_time_str": post_time_str,
            "post_timestamp": post_timestamp,
            "total_replies": reply_count,
            "has_image": has_image,
            "top_lights": top_lights
        }

    except Exception as e:
        logger.error(f"帖子结构解析异常: {e}", exc_info=True)
        return None


# ══════════════════════════════════════════════
#  无限火力主控台
# ══════════════════════════════════════════════
def auto_crawler():
    logger = init_logger(CONFIG["db_name"])
    logger.info("🤖 启动彻夜开火模式，连接浏览器...")

    conn = init_database(CONFIG["db_name"])
    cursor = conn.cursor()

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.pages[0]

        # 统计已有数量
        cursor.execute("SELECT COUNT(*) FROM Posts")
        harvested_count = cursor.fetchone()[0]
        logger.info(f"📊 数据库当前已存有 {harvested_count} 篇帖子。")
        logger.info(f"🚀 本次任务目标：收录满 {CONFIG['max_harvest_posts']} 篇")

        # 【修复】从数据库恢复上次爬到的页码，支持断点续爬
        page_num = load_progress(conn)
        if page_num > 1:
            logger.info(f"🔁 检测到上次进度，从第 {page_num} 页继续...")

        consecutive_errors = 0  # 连续错误计数器

        while harvested_count < CONFIG["max_harvest_posts"]:
            list_url = (
                "https://bbs.hupu.com/topic-daily"
                if page_num == 1
                else f"https://bbs.hupu.com/topic-daily-{page_num}"
            )
            logger.info(f"\n📋 正在横扫列表 第 {page_num} 页...")

            # ── 列表页加载（带重试）──────────────────────
            page_loaded = False
            for attempt in range(1, CONFIG["max_page_retries"] + 1):
                try:
                    page.goto(list_url, timeout=20000)
                    human_like_behavior(page)
                    page_loaded = True
                    consecutive_errors = 0
                    break
                except Exception as e:
                    logger.warning(f"列表页加载失败（第{attempt}次）: {e}")
                    if attempt < CONFIG["max_page_retries"]:
                        random_sleep(CONFIG["delay_on_error"], "错误冷却", logger)
                        try_recover_page(page, logger)

            if not page_loaded:
                logger.error(f"第 {page_num} 页连续 {CONFIG['max_page_retries']} 次失败，跳过此页。")
                consecutive_errors += 1
                if consecutive_errors >= CONFIG["max_consecutive_errors"]:
                    logger.warning(f"连续错误 {consecutive_errors} 次，触发长休眠自保...")
                    random_sleep(CONFIG["long_sleep_on_errors"], "长休眠", logger)
                    consecutive_errors = 0
                page_num += 1
                save_progress(conn, page_num)
                continue

            # ── 提取候选帖子链接（改用 set 去重）─────────
            elements = page.locator("a[href]").all()
            candidates: list[str] = []
            seen: set[str] = set()
            for el in elements:
                href = el.get_attribute("href")
                if href and re.match(r'^/\d+\.html$', href):
                    full_url = "https://bbs.hupu.com" + href
                    if full_url not in seen:          # 【修复】O(1) 集合去重
                        seen.add(full_url)
                        candidates.append(full_url)

            logger.info(f"🎯 本页发现 {len(candidates)} 个候选帖子。")

            # ── 逐帖抓取 ────────────────────────────────
            for url in candidates:
                if harvested_count >= CONFIG["max_harvest_posts"]:
                    break

                # 数据库去重
                cursor.execute("SELECT 1 FROM Posts WHERE url=?", (url,))
                if cursor.fetchone():
                    logger.debug(f"已在库中，跳过: {url}")
                    continue

                logger.info(f"[进度: {harvested_count}/{CONFIG['max_harvest_posts']}] 破门而入: {url}")

                try:
                    page.goto(url, timeout=20000)
                    human_like_behavior(page)

                    post_data = parse_post(page, top_n=CONFIG["top_lights"], url=url, logger=logger)

                    if post_data:
                        cursor.execute('''
                            INSERT INTO Posts (url, title, content, post_time_str, post_timestamp, total_replies, has_image)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            post_data["url"], post_data["title"], post_data["content"],
                            post_data["post_time_str"], post_data["post_timestamp"],
                            post_data["total_replies"], post_data["has_image"]
                        ))

                        for c in post_data["top_lights"]:
                            cursor.execute('''
                                INSERT OR IGNORE INTO Comments
                                    (pid, post_url, username, content, lights, comment_reply_count, comment_timestamp, time_diff_seconds)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                c["pid"], post_data["url"], c["username"], c["content"],
                                c["lights"], c["comment_reply_count"],
                                c["comment_timestamp"], c["time_diff_seconds"]
                            ))

                        conn.commit()
                        harvested_count += 1
                        consecutive_errors = 0
                        logger.info(
                            f"✅ 大丰收｜{post_data['title'][:20]}... "
                            f"| 回复:{post_data['total_replies']} | 亮评:{len(post_data['top_lights'])}条"
                        )

                except Exception as e:
                    logger.error(f"抓取单帖崩溃: {e}", exc_info=True)
                    consecutive_errors += 1

                    # 【新增】连续出错自保机制
                    if consecutive_errors >= CONFIG["max_consecutive_errors"]:
                        logger.warning(f"连续错误 {consecutive_errors} 次，触发长休眠并尝试恢复页面...")
                        random_sleep(CONFIG["long_sleep_on_errors"], "长休眠", logger)
                        try_recover_page(page, logger)
                        consecutive_errors = 0
                    else:
                        random_sleep(CONFIG["delay_on_error"], "错误冷却", logger)
                    continue  # 出错不计入正常休眠，直接下一个

                random_sleep(CONFIG["delay_between_posts"], "正常间隔", logger)

            # ── 翻下一页，持久化进度 ─────────────────────
            page_num += 1
            save_progress(conn, page_num)
            random_sleep(CONFIG["delay_between_pages"], "翻页间隔", logger)

        # ── 任务完成 ─────────────────────────────────────
        reset_progress(conn)  # 清空进度，下次重头开始
        logger.info(f"🎉 彻夜开火任务圆满结束！数据库总量: {harvested_count} 篇。")
        conn.close()


if __name__ == "__main__":
    auto_crawler()