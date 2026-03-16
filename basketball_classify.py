"""
basketball_classify.py — 篮球区帖子 LLM 分类

为每个 section='basketball' 且 ai_tag IS NULL 的帖子打两个维度的标签：
  subject        : 帖子主体（球员/球队/教练/裁判/other）
  event_sentiment: positive / negative / neutral

ai_tag 格式："{subject}-{sentiment}"，例如 "詹姆斯-positive"
无法归类的帖子 ai_tag = "other"

可单独运行：python basketball_classify.py
也可被 main_basketball.py 调用：basketball_classify.run(llm)
"""
import sqlite3
import json
import re
import time
from openai import OpenAI
from config import CONFIG, get_logger

logger = get_logger("BballClassify")

BATCH_SIZE = 10  # 每次 LLM 调用同时处理帖子数（节省 API 调用次数）

_CLASSIFY_SYSTEM = (
    "你是篮球内容分类专家。对每个帖子完成两件事：\n"
    "① 识别主体（帖子的核心话题主角）：\n"
    "   - 球员 → 用标准中文名（如"詹姆斯""库里""科比"，不用英文或绰号）\n"
    "   - 球队 → 标准中文队名（如"湖人""勇士""凯尔特人"）\n"
    "   - 教练 → 输出"教练"\n"
    "   - 裁判 → 输出"裁判"\n"
    "   - 联盟/赛制 → 输出"联盟"\n"
    "   - 无法归类 → 输出"other"\n"
    "② 判断情感倾向：\n"
    "   positive = 高光/胜利/逆转/突破/称赞\n"
    "   negative = 失利/争议/批评/黑料/下课\n"
    "   neutral  = 数据分析/交易/技术讨论/赛程\n"
    "只输出严格 JSON 数组，不要任何前缀或解释：\n"
    '[{"id":1,"subject":"詹姆斯","sentiment":"positive"},{"id":2,"subject":"湖人","sentiment":"negative"}]'
)


def _classify_batch(posts: list[tuple], llm: OpenAI) -> list[dict]:
    """posts: [(url, title, content), ...]  →  [{"url":..., "ai_tag":...}, ...]"""
    lines = []
    for i, (url, title, content) in enumerate(posts):
        snippet = (content or "")[:150].replace("\n", " ")
        lines.append(f'帖子{i+1}：标题=「{title}」 正文=「{snippet}」')

    try:
        resp = llm.chat.completions.create(
            model=CONFIG["model"],
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user",   "content": "\n".join(lines) + "\n请输出分类："},
            ],
            temperature=CONFIG["classify_temperature"],
            max_tokens=500,
            timeout=CONFIG["llm_timeout"],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
        results = json.loads(raw)
        if not isinstance(results, list):
            raise ValueError("返回值不是数组")
    except Exception as e:
        logger.warning(f"分类 LLM 失败（{e}），本批次全标 other")
        results = [{"id": i + 1, "subject": "other", "sentiment": "neutral"}
                   for i in range(len(posts))]

    output = []
    for i, (url, _, _) in enumerate(posts):
        matched = next((r for r in results if r.get("id") == i + 1), None)
        if matched:
            subject   = str(matched.get("subject", "other")).strip()
            sentiment = str(matched.get("sentiment", "neutral")).strip()
            ai_tag = "other" if subject.lower() == "other" else f"{subject}-{sentiment}"
        else:
            ai_tag = "other"
        output.append({"url": url, "ai_tag": ai_tag})
    return output


def run(llm: OpenAI = None) -> int:
    """分类所有未标注的篮球区帖子，返回处理数量。"""
    if llm is None:
        llm = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("""
        SELECT url, title, content FROM Posts
        WHERE section='basketball' AND ai_tag IS NULL
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        logger.info("篮球区没有未分类帖子，跳过")
        return 0

    logger.info(f"🏀 开始分类 {len(rows)} 个篮球区帖子（每批 {BATCH_SIZE} 个）...")
    total = 0
    conn  = sqlite3.connect(CONFIG["db_name"])

    for i in range(0, len(rows), BATCH_SIZE):
        batch   = rows[i : i + BATCH_SIZE]
        results = _classify_batch(batch, llm)
        for r in results:
            conn.execute("UPDATE Posts SET ai_tag=? WHERE url=?", (r["ai_tag"], r["url"]))
            logger.info(f"  [{r['ai_tag']}]")
            total += 1
        conn.commit()
        time.sleep(0.3)  # 避免 API 速率限制

    conn.close()
    logger.info(f"✅ 篮球区分类完成：共 {total} 个帖子")
    return total


if __name__ == "__main__":
    run()
