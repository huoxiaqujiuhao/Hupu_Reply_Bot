"""
rename_tool.py
──────────────
给已打完标签的类别批量重命名，让名字更口语化、更符合虎扑语境。
（此文件原名 debug.py，实际功能是重命名工具）

运行顺序：
  1. test_scraper.py       → 爬虫，采集帖子进库
  2. Taxonomy_generator.py → 聚类，生成 data/taxonomy.json
  3. ai_labeler.py         → 打标，给每个帖子写 ai_tag
  4. rename_tool.py        → 重命名，美化类别名称（可选）
  5. reply_bot.py          → 回复机器人，消费端
"""
import sqlite3
import json
import time
import logging
import os
from openai import OpenAI

os.makedirs("data/logs", exist_ok=True)

# ══════════════════════════════════════════════
#  配置区
# ══════════════════════════════════════════════
CONFIG = {
    "db_name":           "data/hupu_arsenal.db",
    "rename_result_file": "data/rename_result.json",
    "log_file":          "data/logs/rename_tool.log",
    "model":             "qwen3.5-35b-a3b",
    "api_key":           "sk-3d1b473a741d4abba85c9c3fa6933bac",
    "base_url":          "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "sample_size":       15,
    "base_sleep":        1.0,
}


# ══════════════════════════════════════════════
#  日志系统
# ══════════════════════════════════════════════
def init_logger() -> logging.Logger:
    logger = logging.getLogger("Renamer")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(CONFIG["log_file"], encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ══════════════════════════════════════════════
#  数据读取
# ══════════════════════════════════════════════
def get_all_categories(cursor) -> list[str]:
    cursor.execute("""
        SELECT ai_tag FROM Posts
        WHERE ai_tag IS NOT NULL AND ai_tag NOT LIKE 'ERROR%'
        GROUP BY ai_tag ORDER BY COUNT(*) DESC
    """)
    return [row[0] for row in cursor.fetchall()]


def get_sample_posts(cursor, ai_tag: str, n: int) -> list[dict]:
    cursor.execute("""
        SELECT p.url, p.title
        FROM Posts p
        WHERE p.ai_tag = ?
        ORDER BY RANDOM()
        LIMIT ?
    """, (ai_tag, n))
    posts = cursor.fetchall()

    result = []
    for url, title in posts:
        cursor.execute("""
            SELECT content FROM Comments
            WHERE post_url = ?
            ORDER BY lights DESC LIMIT 3
        """, (url,))
        comments = [r[0] for r in cursor.fetchall()]
        result.append({"title": title, "comments": comments})
    return result


# ══════════════════════════════════════════════
#  LLM 命名
# ══════════════════════════════════════════════
def ask_llm_to_rename(client, old_tag: str, samples: list[dict], logger) -> dict | None:
    samples_text = ""
    for i, s in enumerate(samples, 1):
        comments_str = " | ".join(s["comments"]) if s["comments"] else "无"
        samples_text += f"{i}. 标题：{s['title']}\n   高赞评论：{comments_str[:120]}\n\n"

    system_prompt = """你是一个爬虫数据工程师，需要给帖子分类起更通俗易懂的名字。
要求：
- 新名字必须让普通用户一眼看懂，不要学术腔
- 保留"一级类-二级标签"的格式，中间用"-"分隔
- 一级类 2-6 个字，二级标签 2-5 个字
- 用虎扑用户熟悉的语言风格，可以口语化可以玩梗
- 只输出 JSON，不要任何解释

输出格式：
{"primary": "新一级类名", "secondary": "新二级标签", "reason": "一句话说明改名逻辑"}"""

    user_prompt = (
        f"当前类别名：【{old_tag}】\n\n"
        f"该类别下的代表性帖子（共 {len(samples)} 个）：\n\n"
        f"{samples_text}"
        f"请给这个类别起一个更通俗易懂的新名字。"
    )

    try:
        response = client.chat.completions.create(
            model=CONFIG["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            timeout=150,
        )
        result    = json.loads(response.choices[0].message.content.strip())
        primary   = result.get("primary", "").strip()
        secondary = result.get("secondary", "").strip()
        reason    = result.get("reason", "")

        if not primary or not secondary:
            raise ValueError("LLM 返回的名字字段为空")

        return {"new_tag": f"{primary}-{secondary}", "reason": reason}
    except Exception as e:
        logger.error(f"  LLM 命名失败: {e}")
        return None


# ══════════════════════════════════════════════
#  写回数据库
# ══════════════════════════════════════════════
def apply_rename(cursor, conn, old_tag: str, new_tag: str):
    cursor.execute("UPDATE Posts SET ai_tag = ? WHERE ai_tag = ?", (new_tag, old_tag))
    conn.commit()


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════
def main():
    logger = init_logger()
    logger.info("🏷️  类别重命名工具启动")

    conn   = sqlite3.connect(CONFIG["db_name"])
    cursor = conn.cursor()
    client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

    categories = get_all_categories(cursor)
    logger.info(f"📦 共发现 {len(categories)} 个类别，开始逐一重命名...\n" + "═" * 55)

    rename_map = {}
    failed     = []

    for i, old_tag in enumerate(categories, 1):
        cursor.execute("SELECT COUNT(*) FROM Posts WHERE ai_tag = ?", (old_tag,))
        count = cursor.fetchone()[0]
        logger.info(f"\n[{i}/{len(categories)}] 处理：【{old_tag}】（{count} 篇）")

        n       = min(CONFIG["sample_size"], count)
        samples = get_sample_posts(cursor, old_tag, n)
        result  = ask_llm_to_rename(client, old_tag, samples, logger)

        if result:
            new_tag = result["new_tag"]
            reason  = result["reason"]

            if new_tag == old_tag:
                logger.info("  💡 LLM 认为原名已够清晰，保持不变。")
                rename_map[old_tag] = {"new": old_tag, "reason": "保持原名", "count": count}
            else:
                apply_rename(cursor, conn, old_tag, new_tag)
                logger.info(f"  ✅ 改名成功：【{old_tag}】→【{new_tag}】")
                logger.info(f"     理由：{reason}")
                rename_map[old_tag] = {"new": new_tag, "reason": reason, "count": count}
        else:
            logger.warning("  ⚠️ 命名失败，保留原名。")
            failed.append(old_tag)
            rename_map[old_tag] = {"new": old_tag, "reason": "命名失败，保留原名", "count": count}

        time.sleep(CONFIG["base_sleep"])

    with open(CONFIG["rename_result_file"], "w", encoding="utf-8") as f:
        json.dump(rename_map, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'═' * 55}")
    logger.info("📋 改名对照表：")
    logger.info(f"  {'原名':<30} {'新名':<30} {'帖子数'}")
    logger.info(f"  {'-'*70}")
    for old, info in rename_map.items():
        changed = "→" if info["new"] != old else "＝"
        logger.info(f"  {old:<30} {changed} {info['new']:<30} ({info['count']}篇)")

    if failed:
        logger.warning(f"\n⚠️ 以下 {len(failed)} 个类别命名失败，已保留原名：")
        for f in failed:
            logger.warning(f"  - {f}")

    logger.info(f"\n✅ 全部完成！对照表已保存至 {CONFIG['rename_result_file']}")
    conn.close()


if __name__ == "__main__":
    main()