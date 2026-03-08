import sqlite3
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ══════════════════════════════════════════════
#  配置区
# ══════════════════════════════════════════════
CONFIG = {
    "db_name":          "hupu_arsenal.db",
    "taxonomy_file":    "taxonomy.json",
    "model":            "qwen3.5-35b-a3b",  # 如果用122b太慢，可换回qwen-plus
    "api_key":          "sk-3d1b473a741d4abba85c9c3fa6933bac",
    "base_url":         "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",

    "batch_size":       600,   # 你的总帖子才599，直接一步到位，一次性全捞出来！
    "max_workers":      50,    # 开启 50 个真实并发，已经能做到“秒级清屏”
    "request_interval": 0.02,  # 允许极速发包（反正你的 RPM 高达 15000）
    "max_retries":      3,
    "base_sleep":       1.0,
    "content_max_len":  500,    
    "reasoning_max_len": 80,    
}

# ══════════════════════════════════════════════
#  日志系统
# ══════════════════════════════════════════════
def init_logger() -> logging.Logger:
    logger = logging.getLogger("AILabeler")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler("hupu_labeler.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# ══════════════════════════════════════════════
#  加载字典与构建 Prompt
# ══════════════════════════════════════════════
def load_taxonomy_and_prompt() -> tuple[dict, set, str]:
    try:
        with open(CONFIG["taxonomy_file"], "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
    except Exception as e:
        raise RuntimeError(f"读取字典文件失败: {e}")

    valid_primary = set(taxonomy.keys())
    taxonomy_str = json.dumps(taxonomy, ensure_ascii=False, separators=(',', ':'))

    system_prompt = (
        "你是虎扑论坛数据分析师，精通网络黑话。\n"
        "根据帖子标题、正文、高赞评论，从下方菜单选出最匹配的一级和二级分类。\n\n"
        f"菜单：{taxonomy_str}\n\n"
        "输出纯净JSON，格式：\n"
        '{"reasoning":"一句话说明归类理由，严格不超过80字","primary_category":"一级分类名","secondary_tag":"二级标签名"}'
    )
    return taxonomy, valid_primary, system_prompt

# ══════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════
def get_top_comments(cursor, post_url: str, limit: int = 3) -> str:
    cursor.execute("SELECT content FROM Comments WHERE post_url=? ORDER BY lights DESC LIMIT ?", (post_url, limit))
    comments = cursor.fetchall()
    return " | ".join([f"{i+1}.{c[0]}" for i, c in enumerate(comments)]) if comments else "无"

def truncate_content(content: str) -> str:
    if not content: return "无"
    return content[:CONFIG["content_max_len"]] + "…" if len(content) > CONFIG["content_max_len"] else content

def truncate_reasoning(reasoning: str) -> str:
    if not reasoning: return ""
    return reasoning[:CONFIG["reasoning_max_len"]] + "…" if len(reasoning) > CONFIG["reasoning_max_len"] else reasoning

# ══════════════════════════════════════════════
#  API 请求核心（线程内执行）
# ══════════════════════════════════════════════
_rate_lock = threading.Lock()
_last_request_time = 0.0

def call_ai_with_retry(client, title, content, comments_str, system_prompt, taxonomy, valid_primary, logger):
    user_prompt = f"标题:{title}\n正文:{truncate_content(content)}\n评论:{comments_str}"

    for attempt in range(1, CONFIG["max_retries"] + 1):
        global _last_request_time
        with _rate_lock:
            now = time.time()
            gap = CONFIG["request_interval"] - (now - _last_request_time)
            if gap > 0: time.sleep(gap)
            _last_request_time = time.time()

        try:
            response = client.chat.completions.create(
                model=CONFIG["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=120 # 给稍微充裕一点的超时时间
            )

            result_json = json.loads(response.choices[0].message.content.strip())
            primary   = result_json.get("primary_category", "")
            secondary = result_json.get("secondary_tag", "")
            reasoning = result_json.get("reasoning", "")

            if primary not in valid_primary: raise ValueError(f"幻觉拦截: 捏造一级 [{primary}]")
            if secondary not in taxonomy[primary]: raise ValueError(f"幻觉拦截: 捏造二级 [{secondary}]")

            return f"{primary}-{secondary}", truncate_reasoning(reasoning)

        except Exception as e:
            wait_time = CONFIG["base_sleep"] * (2 ** (attempt - 1))
            logger.debug(f"  [重试 {attempt}/{CONFIG['max_retries']}] {e} → 等 {wait_time}s")
            time.sleep(wait_time)

    return None, None

# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════
def flush_results_to_db(cursor, conn, results_buffer, logger, global_processed, total_remaining):
    """提取的辅助函数：将缓存结果写入数据库"""
    if not results_buffer: return 0
    
    success_batch = 0
    for url, title, final_tag, reasoning in results_buffer:
        if final_tag:
            cursor.execute("UPDATE Posts SET ai_tag=?, ai_reasoning=? WHERE url=?", (final_tag, reasoning, url))
            success_batch += 1
            logger.info(f"  ✅ [{global_processed}/{total_remaining}] {title[:15]}... ➜ {final_tag}")
        else:
            # 💡 拆除 SQLite 炸弹的核心：标记为【彻底失败】，下次查询就不会捞它了
            cursor.execute("UPDATE Posts SET ai_tag='ERROR-彻底失败' WHERE url=?", (url,))
            logger.warning(f"  ⚠️ {title[:15]}... 多次失败，标记为[ERROR-彻底失败]并放弃。")
            
    conn.commit()
    results_buffer.clear()
    return success_batch

def main():
    logger = init_logger()
    logger.info("🚀 虎扑 AI 打标厂（防弹可刹车版）引擎点火！")

    taxonomy, valid_primary, system_prompt = load_taxonomy_and_prompt()
    
    conn = sqlite3.connect(CONFIG["db_name"])
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE Posts ADD COLUMN ai_reasoning TEXT")
        conn.commit()
    except sqlite3.OperationalError: pass

    client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

    cursor.execute("SELECT COUNT(*) FROM Posts WHERE ai_tag IS NULL OR ai_tag = 'ERROR-打标失败'")
    total_remaining = cursor.fetchone()[0]

    if total_remaining == 0:
        logger.info("🍻 所有帖子均已打标完毕，下班！")
        conn.close(); return

    logger.info(f"📦 剩余任务: {total_remaining} 个 | 并发线程: {CONFIG['max_workers']} | 正文截断: {CONFIG['content_max_len']}字\n" + "═" * 55)

    success_count = 0
    global_processed = 0
    results_buffer = []

    # 声明线程池
    executor = ThreadPoolExecutor(max_workers=CONFIG["max_workers"])

    try:
        while True:
            # 💡 极简 SQL：不再使用危险的 NOT IN
            cursor.execute(
                "SELECT url, title, content FROM Posts "
                "WHERE ai_tag IS NULL OR ai_tag = 'ERROR-打标失败' LIMIT ?",
                (CONFIG["batch_size"],)
            )
            batch_posts = cursor.fetchall()
            
            if not batch_posts: break

            tasks = []
            for url, title, content in batch_posts:
                comments_str = get_top_comments(cursor, url, limit=3)
                tasks.append((url, title, content, comments_str))

            logger.info(f"\n🌊 抛出 {len(tasks)} 个并发请求...")
            
            future_map = {
                executor.submit(
                    call_ai_with_retry, client, t[1], t[2], t[3],
                    system_prompt, taxonomy, valid_primary, logger
                ): (t[0], t[1])
                for t in tasks
            }

            for future in as_completed(future_map):
                url, title = future_map[future]
                global_processed += 1
                try:
                    final_tag, reasoning = future.result()
                    results_buffer.append((url, title, final_tag, reasoning))
                except Exception as e:
                    logger.error(f"  ❌ 线程执行异常 [{title[:15]}]: {e}")
                    results_buffer.append((url, title, None, None))

            # 本批次全部请求完后，集中写库
            logger.info(f"💾 正在写入本批 {len(results_buffer)} 条结果...")
            success_count += flush_results_to_db(cursor, conn, results_buffer, logger, global_processed, total_remaining)
            time.sleep(CONFIG["base_sleep"])

    except KeyboardInterrupt:
        # 🛑 核心刹车系统生效！
        logger.warning("\n\n🛑================ 紧急刹车 ================")
        logger.warning("收到 Ctrl+C 中断指令！正在取消未派发的任务，请勿关闭终端...")
        
        # 取消所有还在排队、没来得及发出的请求
        executor.shutdown(wait=False, cancel_futures=True) 
        
        # 把内存里已经跑完的结果赶紧存进硬盘
        if results_buffer:
            logger.warning(f"💾 正在抢救缓存中的 {len(results_buffer)} 条数据写入数据库...")
            success_count += flush_results_to_db(cursor, conn, results_buffer, logger, global_processed, total_remaining)
        logger.warning("🛑 刹车完成，安全退出程序！\n")

    finally:
        if not results_buffer: # 正常跑完的情况
            logger.info(f"\n{'═' * 55}\n🎉 本次运行结束！成功打标 {success_count} 个帖子。")
        conn.close()

if __name__ == "__main__":
    main()