"""
profile_scraper.py — 回收 Bot 评论的点赞反馈
════════════════════════════════════════════
功能：
  1. 调用虎扑个人主页 API，拉取 Bot 发出的所有历史评论（含点赞数）
  2. 结果写入 SQLite 的 BotComments 表
  3. 支持增量更新：已抓过的 pid 只更新点赞数，不重复插入

运行方式：
  python profile_scraper.py              # 全量拉取
  python profile_scraper.py --update     # 只更新近期评论的点赞数（快）
"""
import sqlite3
import json
import time
import argparse
import os
from datetime import datetime, timezone
from playwright.sync_api import sync_playwright
from config import CONFIG, get_logger

logger = get_logger("ProfileScraper")

# ── 目标账号 ─────────────────────────────────────────────
EUID      = "168368498719798"
PAGE_SIZE = 20
API_BASE  = (
    f"https://my.hupu.com/pcmapi/pc/space/v1/getReplyList"
    f"?euid={EUID}&pageSize={PAGE_SIZE}"
)


# ══════════════════════════════════════════════
#  数据库初始化
# ══════════════════════════════════════════════
def init_db() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(CONFIG["db_name"])
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS BotComments (
            pid              INTEGER PRIMARY KEY,
            tid              INTEGER,
            post_url         TEXT,
            content          TEXT,
            created_at       INTEGER,
            light_count      INTEGER DEFAULT 0,
            unlight_count    INTEGER DEFAULT 0,
            score            INTEGER DEFAULT 0,
            checked_at       INTEGER
        )
    """)
    conn.commit()
    return conn


# ══════════════════════════════════════════════
#  API 请求（借用浏览器的登录 Cookie）
# ══════════════════════════════════════════════
def fetch_reply_page(context, max_time: int, page_num: int) -> dict | None:
    url = f"{API_BASE}&maxTime={max_time}&page={page_num}"
    try:
        resp = context.request.get(url, timeout=15000)
        if resp.status == 200:
            return resp.json()
        else:
            logger.warning(f"HTTP {resp.status}: {url}")
            return None
    except Exception as e:
        logger.error(f"请求失败: {e}")
        return None


# ══════════════════════════════════════════════
#  全量拉取
# ══════════════════════════════════════════════
def full_fetch(conn: sqlite3.Connection, context):
    """从最新评论往前翻，直到没有数据为止"""
    cur         = conn.cursor()
    now_ts      = int(datetime.now(timezone.utc).timestamp())
    max_time    = now_ts
    page_num    = 1
    total       = 0
    checked_at  = now_ts

    logger.info("🔄 开始全量拉取评论历史...")

    while True:
        data = fetch_reply_page(context, max_time, page_num)
        if not data or data.get("code") != 1:
            logger.warning(f"API 返回异常: {data}")
            break

        items = data.get("data", {}).get("replyWithQuoteDtoList", [])
        if not items:
            logger.info("已到最后一页，拉取完毕")
            break

        for item in items:
            pid           = item.get("pid")
            tid           = item.get("tid")
            post_url      = f"https://bbs.hupu.com/{tid}.html" if tid else None
            content       = item.get("content", "")
            created_at    = item.get("createTime", 0)
            light_count   = item.get("lightCount", 0)
            unlight_count = item.get("unlightCount", 0)
            score         = item.get("score", 0)

            cur.execute("""
                INSERT INTO BotComments
                    (pid, tid, post_url, content, created_at,
                     light_count, unlight_count, score, checked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pid) DO UPDATE SET
                    light_count   = excluded.light_count,
                    unlight_count = excluded.unlight_count,
                    score         = excluded.score,
                    checked_at    = excluded.checked_at
            """, (pid, tid, post_url, content, created_at,
                  light_count, unlight_count, score, checked_at))

        conn.commit()
        total += len(items)

        # 翻页：maxTime 取本页最后一条的 createTime
        last_item = items[-1]
        max_time  = last_item.get("createTime", 0)
        page_num += 1

        logger.info(
            f"  第 {page_num-1} 页 {len(items)} 条 | "
            f"最早: {datetime.fromtimestamp(max_time).strftime('%m-%d %H:%M')} | "
            f"累计: {total} 条"
        )
        time.sleep(0.5)  # 礼貌性延迟

    logger.info(f"✅ 全量拉取完成，共 {total} 条评论写入数据库")
    return total


# ══════════════════════════════════════════════
#  增量更新（只刷近期点赞数）
# ══════════════════════════════════════════════
def incremental_update(conn: sqlite3.Connection, context, days: int = 3):
    """只更新最近 N 天内发出的评论的点赞数"""
    cur        = conn.cursor()
    since_ts   = int(datetime.now(timezone.utc).timestamp()) - days * 86400
    checked_at = int(datetime.now(timezone.utc).timestamp())

    # 从数据库找出近期评论的 pid
    cur.execute("""
        SELECT pid, post_url, content, created_at
        FROM BotComments
        WHERE created_at >= ?
        ORDER BY created_at DESC
    """, (since_ts,))
    recent = cur.fetchall()

    if not recent:
        logger.info(f"近 {days} 天内没有评论记录")
        return

    logger.info(f"🔄 增量更新：近 {days} 天共 {len(recent)} 条评论，重新拉取点赞数...")

    # 重新拉最近几页覆盖更新
    now_ts   = int(datetime.now(timezone.utc).timestamp())
    max_time = now_ts
    page_num = 1
    updated  = 0

    # 找出需要更新的 pid 集合
    target_pids = {row[0] for row in recent}

    while target_pids:
        data = fetch_reply_page(context, max_time, page_num)
        if not data or data.get("code") != 1:
            break

        items = data.get("data", {}).get("replyWithQuoteDtoList", [])
        if not items:
            break

        for item in items:
            pid = item.get("pid")
            if pid in target_pids:
                cur.execute("""
                    UPDATE BotComments
                    SET light_count=?, unlight_count=?, score=?, checked_at=?
                    WHERE pid=?
                """, (
                    item.get("lightCount", 0),
                    item.get("unlightCount", 0),
                    item.get("score", 0),
                    checked_at,
                    pid,
                ))
                target_pids.discard(pid)
                updated += 1

        conn.commit()
        last_item = items[-1]
        # 如果最后一条已超出时间范围就停
        if last_item.get("createTime", 0) < since_ts:
            break

        max_time  = last_item.get("createTime", 0)
        page_num += 1
        time.sleep(0.5)

    logger.info(f"✅ 增量更新完成，更新了 {updated} 条评论的点赞数")


# ══════════════════════════════════════════════
#  统计摘要
# ══════════════════════════════════════════════
def print_summary(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM BotComments")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM BotComments WHERE light_count >= 3")
    high_likes = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM BotComments WHERE light_count = 0")
    zero_likes = cur.fetchone()[0]

    cur.execute("SELECT MAX(light_count), content, post_url FROM BotComments")
    best = cur.fetchone()

    logger.info(f"\n{'='*55}")
    logger.info(f"📊 数据库摘要：")
    logger.info(f"   总评论数:       {total}")
    logger.info(f"   点赞 ≥ 3 的:    {high_likes} 条")
    logger.info(f"   点赞为 0 的:    {zero_likes} 条")
    if best and best[0]:
        logger.info(f"   最高点赞:       {best[0]} 👍")
        logger.info(f"   内容:           {str(best[1])[:50]}")
        logger.info(f"   帖子:           {best[2]}")
    logger.info(f"{'='*55}\n")


# ══════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="只更新近期点赞数（不全量拉取）")
    parser.add_argument("--days", type=int, default=3,
                        help="--update 模式下回溯天数（默认3天）")
    args = parser.parse_args()

    conn = init_db()

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]

        if args.update:
            incremental_update(conn, context, days=args.days)
        else:
            full_fetch(conn, context)

        print_summary(conn)

    conn.close()


if __name__ == "__main__":
    main()