"""
main.py — 一键启动总控台
════════════════════════════════════════
运行方式：python main.py

每轮循环：
  爬虫 (scrape_minutes) → 分类 → [other重聚类] → 回复 (reply_minutes) → 重复

急行军逻辑：
  整个 session 里回复时间是有预算的（total_reply_budget）。
  当累计回复时间 / 总预算 超过 panic_ratio 时，切换为急行军模式。
  例：2h session，每轮20min回复 → 总预算≈40min，
      panic_ratio=0.75 → 前30min从容，最后10min急行军。 
════════════════════════════════════════
"""
import os
os.environ["HF_HOME"] = "E:/models/huggingface"

import time
import socket
import subprocess
import sqlite3
import logging
from logging.handlers import TimedRotatingFileHandler
from config import CONFIG, get_logger

os.makedirs("data/logs", exist_ok=True)
logger = get_logger("Main")


# ══════════════════════════════════════════════
#  按日期滚动日志（每天一个文件，保留 7 天）
# ══════════════════════════════════════════════
def _setup_rotating_log():
    """把 config.py 里的 FileHandler 替换成滚动版本"""
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] [%(name)-11s] %(message)s",
        datefmt="%H:%M:%S",
    )
    rotating = TimedRotatingFileHandler(
        "data/logs/hupu.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    rotating.setLevel(logging.DEBUG)
    rotating.setFormatter(fmt)

    # 遍历所有已注册的 logger，把 FileHandler 换成滚动版
    for lgr in logging.Logger.manager.loggerDict.values():
        if not isinstance(lgr, logging.Logger):
            continue
        for h in lgr.handlers[:]:
            if isinstance(h, logging.FileHandler) and not isinstance(h, TimedRotatingFileHandler):
                lgr.removeHandler(h)
                h.close()
                lgr.addHandler(rotating)

_setup_rotating_log()


# ══════════════════════════════════════════════
#  Chrome 自动拉起
# ══════════════════════════════════════════════
CHROME_PROFILE = os.path.abspath("data/chrome_profile")
CDP_PORT = 9222

CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

def _cdp_alive() -> bool:
    """检查 CDP 端口是否已在监听"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(("127.0.0.1", CDP_PORT))
        s.close()
        return result == 0
    except Exception:
        return False


def ensure_chrome():
    """
    确保 Chrome 以 CDP 模式运行，并确认已登录。
    - 若端口已开：复用现有会话，但仍检查登录状态
    - 若未开：启动 Chrome，检查登录状态
    Cookie 保存在 data/chrome_profile，登录一次永久有效。
    """
    if _cdp_alive():
        logger.info("✅ Chrome CDP 已就绪，复用现有会话")
    else:
        chrome_exe = None
        for p in CHROME_PATHS:
            if os.path.exists(p):
                chrome_exe = p
                break

        if not chrome_exe:
            logger.error("❌ 未找到 Chrome，请手动启动并打开 CDP 端口后按回车")
            input()
            return

        logger.info("🌐 启动 Chrome（专属 Profile，保留登录状态）...")
        os.makedirs(CHROME_PROFILE, exist_ok=True)
        subprocess.Popen([
            chrome_exe,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={CHROME_PROFILE}",
            "--no-first-run",
            "--no-default-browser-check",
            "https://bbs.hupu.com",
        ])

        # 等 Chrome 启动
        for _ in range(20):
            if _cdp_alive():
                break
            time.sleep(0.5)

        if not _cdp_alive():
            logger.warning("Chrome 启动超时，请手动确认")

    # 无论是复用还是新启动，都要求用户手动确认已登录
    # （Cookie 文件存在不等于 session 有效，不做自动判断）
    print("\n" + "=" * 60)
    print("  程序已暂停。")
    print()
    print("  请确认 Chrome 里虎扑已登录：")
    print("    - 已登录 → 直接按回车，程序立刻开始运行")
    print("    - 未登录 → 先登录，再回来按回车")
    print("=" * 60)
    input("  > 按回车继续：")
    logger.info("✅ 用户确认登录完成，继续启动")


# ══════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════
def count_untagged() -> int:
    try:
        conn = sqlite3.connect(CONFIG["db_name"])
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Posts WHERE ai_tag IS NULL")
        n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


def count_tagged() -> int:
    try:
        conn = sqlite3.connect(CONFIG["db_name"])
        cur  = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM Posts
            WHERE ai_tag IS NOT NULL
              AND ai_tag NOT LIKE 'ERROR%'
              AND ai_tag != 'other'
        """)
        n = cur.fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


def compute_total_reply_budget(total_seconds: float) -> float:
    """整个 session 里分配给回复的总秒数"""
    cycle_total   = CONFIG["scrape_minutes"] + CONFIG["reply_minutes"]
    reply_fraction = CONFIG["reply_minutes"] / cycle_total
    return total_seconds * reply_fraction


# ══════════════════════════════════════════════
#  主控
# ══════════════════════════════════════════════
def main():
    ensure_chrome()

    program_start = time.time()
    total_seconds = CONFIG["total_max_hours"] * 3600

    logger.info("=" * 65)
    logger.info("🚀 一键启动！")
    logger.info(f"   总时长: {CONFIG['total_max_hours']}h  |  "
                f"爬虫: {CONFIG['scrape_minutes']}min/轮  |  "
                f"回复: {CONFIG['reply_minutes']}min/轮")
    logger.info(f"   急行军触发：累计回复时间超过总预算的 {CONFIG['panic_ratio']*100:.0f}%")
    logger.info("=" * 65)

    # ── 全局初始化（只做一次）────────────────────
    logger.info("🧠 加载 Embedding 模型（首次运行稍慢）...")
    from embedder import EmbeddingModel
    from openai import OpenAI
    import scraper as test_scraper
    import classifier
    import reply_bot
    import basketball_reply
    import profile_scraper
    import memory_reviewer

    emb_model = EmbeddingModel(CONFIG["embedding_model"])
    llm       = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

    # 篮球区：构建 tier 表 + 篮球专属向量库 + 启动后台扫描线程
    basketball_reply.build_tier_map()
    basketball_reply.build_bball_vector_store(emb_model)
    _bball_scanner = basketball_reply.BballScanner()
    _bball_scanner.start()
    logger.info("✅ 初始化完成\n")

    # ── 启动前：刷新点赞 + 反思 ──────────────────────────
    logger.info("\n🔄 【启动前置】刷新历史评论点赞数...")
    try:
        profile_scraper.incremental_update_main()
    except Exception as e:
        logger.error(f"点赞刷新出错: {e}", exc_info=True)

    logger.info("\n🧠 【启动前置】反思历史评论...")
    try:
        memory_reviewer.run(emb_model=emb_model, llm=llm)
    except Exception as e:
        logger.error(f"反思流程出错: {e}", exc_info=True)

    # 如果质心文件不存在，从数据库重建（利用已有的打标数据）
    if not os.path.exists(CONFIG["centroids_file"]):
        logger.info("📐 首次运行，从数据库重建质心...")
        classifier.rebuild_centroids(emb_model)

    # 累计回复时间跟踪（本次启动内跨轮次累加）
    cum_reply_secs    = 0.0
    total_reply_budget = compute_total_reply_budget(total_seconds)

    logger.info(f"📊 总回复预算: {total_reply_budget/60:.1f} 分钟 "
                f"（急行军阈值: {total_reply_budget * CONFIG['panic_ratio'] / 60:.1f} 分钟）\n")

    cycle = 1

    while True:
        elapsed   = time.time() - program_start
        remaining = total_seconds - elapsed

        if remaining <= 60:
            logger.info("⏰ 剩余时间不足 1 分钟，结束。")
            break

        logger.info(f"\n{'='*65}")
        logger.info(f"🔄 第 {cycle} 轮  |  剩余: {remaining/3600:.2f}h  |  "
                    f"累计回复: {cum_reply_secs/60:.1f}/{total_reply_budget/60:.1f} min")
        logger.info(f"{'='*65}")

        # 本轮共享的已回复 URL 集合（爬虫和回复两阶段都用）
        cycle_replied_urls: set = set()

        # ────────────────────────────────────────────
        #  阶段一：爬虫（同时消费篮球队列）
        # ────────────────────────────────────────────
        scrape_budget   = min(CONFIG["scrape_minutes"] * 60, remaining)
        scrape_deadline = time.time() + scrape_budget

        logger.info(f"\n🕷️  【爬虫阶段】时长 {scrape_budget/60:.0f} 分钟")
        try:
            def bball_drain_scrape(page, replied_urls):
                return basketball_reply.drain_bball_queue(
                    page, replied_urls, emb_model, llm
                )

            test_scraper.auto_crawler(
                deadline=scrape_deadline,
                bball_drain_fn=bball_drain_scrape,
                replied_urls=cycle_replied_urls,
            )
        except Exception as e:
            logger.error(f"爬虫出错: {e}", exc_info=True)

        elapsed = time.time() - program_start
        if total_seconds - elapsed <= 60:
            break

        # ────────────────────────────────────────────
        #  阶段二：分类
        # ────────────────────────────────────────────
        untagged = count_untagged()
        logger.info(f"\n🏷️  【分类阶段】未标注帖子: {untagged} 个")

        if untagged > 0:
            try:
                classified = classifier.classify_posts(emb_model)
                logger.info(f"   本次分类处理: {classified} 个帖子")

                # 检查 other 是否需要重聚类
                triggered = classifier.check_and_recluster_others(emb_model, llm)
                if triggered:
                    logger.info("   ✅ 重聚类完成，新类别已就绪，本轮回复可立即使用")
            except Exception as e:
                logger.error(f"分类出错: {e}", exc_info=True)
        else:
            logger.info("   无需分类，跳过")

        elapsed = time.time() - program_start
        if total_seconds - elapsed <= 60:
            break

        # ────────────────────────────────────────────
        #  阶段三：回复
        # ────────────────────────────────────────────
        tagged_count  = count_tagged()
        reply_budget  = min(CONFIG["reply_minutes"] * 60,
                            total_seconds - (time.time() - program_start))
        reply_deadline = time.time() + reply_budget

        # 当前急行军状态（基于累计时间）
        panic_threshold = total_reply_budget * CONFIG["panic_ratio"]
        already_panic   = cum_reply_secs >= panic_threshold

        logger.info(f"\n💬 【回复阶段】时长 {reply_budget/60:.0f} 分钟  |  "
                    f"已打标帖子: {tagged_count} 个  |  "
                    f"当前初始模式: {'🚨 急行军' if already_panic else '😌 从容'}")

        if tagged_count < 10:
            logger.warning("   ⚠️ 打标帖子不足 10 个，回复质量可能较差，继续执行")

        reply_phase_start = time.time()
        try:
            vector_store = reply_bot.InMemoryVectorStore(CONFIG["db_name"], emb_model)

            def bball_drain(page, replied_urls):
                return basketball_reply.drain_bball_queue(
                    page, replied_urls, emb_model, llm
                )

            reply_bot.auto_crawler(
                deadline=reply_deadline,
                cum_reply_secs_at_start=cum_reply_secs,
                total_reply_budget=total_reply_budget,
                vector_store=vector_store,
                emb_model=emb_model,
                llm=llm,
                bball_drain_fn=bball_drain,
                replied_urls_init=cycle_replied_urls,
            )
        except Exception as e:
            logger.error(f"回复出错: {e}", exc_info=True)

        # 累加本轮实际回复时长
        cum_reply_secs += (time.time() - reply_phase_start)
        logger.info(f"   累计回复时间: {cum_reply_secs/60:.1f} min / "
                    f"预算 {total_reply_budget/60:.1f} min")

        elapsed = time.time() - program_start
        if total_seconds - elapsed <= 60:
            break

        cycle += 1

    total_used = (time.time() - program_start) / 3600
    logger.info(f"\n{'='*65}")
    logger.info(f"✅ 程序结束！总运行时长: {total_used:.2f}h | "
                f"共 {cycle} 轮 | 累计回复: {cum_reply_secs/60:.1f} min")
    logger.info(f"{'='*65}")


if __name__ == "__main__":
    main()