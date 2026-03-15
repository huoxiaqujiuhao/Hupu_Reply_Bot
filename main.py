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
import time
import os
import sqlite3
from config import CONFIG, get_logger

os.makedirs("data/logs", exist_ok=True)
logger = get_logger("Main")


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
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI
    import scraper as test_scraper
    import classifier
    import reply_bot
    import profile_scraper
    import memory_reviewer

    emb_model = SentenceTransformer(CONFIG["embedding_model"])
    llm       = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])
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

        # ────────────────────────────────────────────
        #  阶段一：爬虫
        # ────────────────────────────────────────────
        scrape_budget   = min(CONFIG["scrape_minutes"] * 60, remaining)
        scrape_deadline = time.time() + scrape_budget

        logger.info(f"\n🕷️  【爬虫阶段】时长 {scrape_budget/60:.0f} 分钟")
        try:
            test_scraper.auto_crawler(deadline=scrape_deadline)
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
            reply_bot.auto_crawler(
                deadline=reply_deadline,
                cum_reply_secs_at_start=cum_reply_secs,
                total_reply_budget=total_reply_budget,
                vector_store=vector_store,
                emb_model=emb_model,
                llm=llm,
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