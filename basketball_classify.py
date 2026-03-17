# -*- coding: utf-8 -*-
"""
basketball_classify.py - basketball post LLM classifier

ai_tag format: "subject-sentiment", e.g. "詹姆斯-positive"
Unclassifiable posts get ai_tag = "other"
"""
import sqlite3
import json
import re
import time
from openai import OpenAI
from config import CONFIG, get_logger

logger = get_logger("BballClassify")

BATCH_SIZE = 10


def _classify_batch(posts, llm):
    lines = []
    for i, (url, title, content) in enumerate(posts):
        snippet = (content or "")[:150].replace("\n", " ")
        lines.append("post{}: title={} body={}".format(i + 1, title, snippet))

    try:
        resp = llm.chat.completions.create(
            model=CONFIG["model"],
            messages=[
                {"role": "system", "content": CONFIG["prompt_basketball_classify_system"]},
                {"role": "user",   "content": "\n".join(lines) + "\n请输出分类JSON："},
            ],
            temperature=CONFIG["classify_temperature"],
            max_tokens=500,
            timeout=CONFIG["llm_timeout"],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
        results = json.loads(raw)
        if not isinstance(results, list):
            raise ValueError("not a list")
    except Exception as e:
        logger.warning("classify LLM failed ({}), marking batch as other".format(e))
        results = [{"id": i + 1, "subject": "other", "sentiment": "neutral"}
                   for i in range(len(posts))]

    output = []
    for i, (url, _, _) in enumerate(posts):
        matched = next((r for r in results if r.get("id") == i + 1), None)
        if matched:
            subject   = str(matched.get("subject", "other")).strip()
            sentiment = str(matched.get("sentiment", "neutral")).strip()
            ai_tag = "other" if subject.lower() == "other" else "{}-{}".format(subject, sentiment)
        else:
            ai_tag = "other"
        output.append({"url": url, "ai_tag": ai_tag})
    return output


def run(llm=None):
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
        logger.info("no unclassified basketball posts")
        return 0

    logger.info("classifying {} basketball posts (batch={})".format(len(rows), BATCH_SIZE))
    total = 0
    conn  = sqlite3.connect(CONFIG["db_name"])

    for i in range(0, len(rows), BATCH_SIZE):
        batch   = rows[i: i + BATCH_SIZE]
        results = _classify_batch(batch, llm)
        for r in results:
            conn.execute("UPDATE Posts SET ai_tag=? WHERE url=?", (r["ai_tag"], r["url"]))
            logger.info("  [{}]".format(r["ai_tag"]))
            total += 1
        conn.commit()
        time.sleep(0.3)

    conn.close()
    logger.info("done: {} posts classified".format(total))
    return total


if __name__ == "__main__":
    run()
