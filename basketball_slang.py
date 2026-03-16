"""
basketball_slang.py — 篮球区球星/球队黑话提炼 + 自动更新

流程：
  1. 从 Comments 表收集分类好的篮球帖的高赞评论（按 subject 分组）
  2. 调用 LLM 提炼每个 subject 的球迷黑话（支持者/反对者/中立）
  3. 向量化后存入 SlangTerms 表
  4. 自动更新：当某 subject 的新评论数超过上次提炼时的 1.5 倍，重新提炼

可单独运行：python basketball_slang.py
也可被 main_basketball.py 调用：basketball_slang.run(emb_model, llm)
"""
import sqlite3
import json
import re
import time
import numpy as np
from openai import OpenAI
from embedder import EmbeddingModel
from config import CONFIG, get_logger
import memory_store

logger = get_logger("BballSlang")

SLANG_MIN_LIGHTS     = 15    # 评论至少多少赞才被纳入提炼素材
MAX_COMMENTS_PER_RUN = 60    # 每个 subject 每次最多喂给 LLM 多少条评论
BATCH_SIZE           = 30    # 大 subject 分批提炼时每批评论数
UPDATE_GROWTH_RATIO  = 1.5   # 新评论 > 上次数量 × 此倍数 → 触发重新提炼

_SLANG_SYSTEM = (
    "你是虎扑篮球区的资深用户和语言分析师。\n"
    "我会给你一批关于【{subject}】的高赞评论，请提炼球迷常用的黑话、称谓、表达模式。\n"
    "分三类：\n"
    "  fan（支持者/粉丝）    ：他们如何称呼他、常用哪些正面词/绰号\n"
    "  hater（黑子/反对者）  ：他们如何称呼他、常用哪些负面词/梗\n"
    "  neutral（中立/客观）  ：数据类、历史类、客观描述词汇\n"
    "要求：每个 term 1-6 字，desc 简短说明背景（可为空字符串）\n"
    '只输出 JSON：{{"fan":[{{"term":"...","desc":"..."}}],"hater":[...],"neutral":[...]}}'
)


# ══════════════════════════════════════════════
#  数据库工具
# ══════════════════════════════════════════════
def _init_slang_table():
    """确保 SlangTerms 表存在（已在 memory_store.init_db 里建，这里做保险）"""
    memory_store.init_db()


def _get_subjects(conn) -> list[str]:
    """返回所有已分类的篮球区帖子的 subject 列表（排除 other）"""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ai_tag FROM Posts
        WHERE section='basketball'
          AND ai_tag IS NOT NULL
          AND ai_tag != 'other'
    """)
    subjects = set()
    for (tag,) in cur.fetchall():
        if '-' in tag:
            subject = tag.rsplit('-', 1)[0]
            subjects.add(subject)
    return sorted(subjects)


def _get_comments_for_subject(subject: str, conn) -> list[str]:
    """获取某 subject 所有帖子的高赞评论文本"""
    cur = conn.cursor()
    cur.execute("""
        SELECT c.content, c.lights
        FROM Comments c
        JOIN Posts p ON c.post_url = p.url
        WHERE p.section = 'basketball'
          AND p.ai_tag LIKE ?
          AND c.lights >= ?
        ORDER BY c.lights DESC
        LIMIT ?
    """, (f"{subject}-%", SLANG_MIN_LIGHTS, MAX_COMMENTS_PER_RUN))
    rows = cur.fetchall()
    return [content for content, _ in rows], len(rows)


def _get_comment_count(subject: str, conn) -> int:
    """获取某 subject 当前高赞评论总数（用于判断是否需要更新）"""
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM Comments c
        JOIN Posts p ON c.post_url = p.url
        WHERE p.section='basketball'
          AND p.ai_tag LIKE ?
          AND c.lights >= ?
    """, (f"{subject}-%", SLANG_MIN_LIGHTS))
    return cur.fetchone()[0]


def _get_stored_comment_count(subject: str, conn) -> int:
    """获取上次提炼时记录的评论数"""
    cur = conn.cursor()
    cur.execute(
        "SELECT MAX(source_comment_count) FROM SlangTerms WHERE subject=?",
        (subject,)
    )
    row = cur.fetchone()
    return row[0] if row and row[0] else 0


def _needs_update(subject: str, conn) -> tuple[bool, int]:
    """返回 (是否需要更新, 当前评论数)"""
    current = _get_comment_count(subject, conn)
    if current < 5:                         # 素材太少，不提炼
        return False, current
    stored = _get_stored_comment_count(subject, conn)
    if stored == 0:                         # 从未提炼过
        return True, current
    if current > stored * UPDATE_GROWTH_RATIO:  # 新增了足够多的素材
        return True, current
    return False, current


# ══════════════════════════════════════════════
#  LLM 提炼
# ══════════════════════════════════════════════
def _extract_slang_batch(subject: str, comments: list[str], llm: OpenAI) -> dict:
    """对单批评论调用 LLM 提炼黑话，返回 {fan:[...], hater:[...], neutral:[...]}"""
    comments_text = "\n".join(
        f"{i+1}. {c[:120]}" for i, c in enumerate(comments)
    )
    system = _SLANG_SYSTEM.format(subject=subject)
    resp = llm.chat.completions.create(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"以下是关于【{subject}】的高赞评论：\n{comments_text}\n\n请提炼黑话："},
        ],
        temperature=0.3,
        max_tokens=1200,
        timeout=CONFIG["llm_timeout"],
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r'^```[a-z]*\n?', '', raw).rstrip('`').strip()
    return json.loads(raw)


def _extract_slang(subject: str, comments: list[str], llm: OpenAI) -> dict:
    """调用 LLM 提炼黑话，支持大 subject 分批合并，返回 {fan:[...], hater:[...], neutral:[...]}"""
    merged = {"fan": [], "hater": [], "neutral": []}

    # 分批处理，避免超出 max_tokens
    for start in range(0, len(comments), BATCH_SIZE):
        batch = comments[start: start + BATCH_SIZE]
        try:
            result = _extract_slang_batch(subject, batch, llm)
            for stance in ("fan", "hater", "neutral"):
                merged[stance].extend(result.get(stance, []))
        except Exception as e:
            logger.warning(f"黑话提炼 LLM 失败（{subject} batch {start//BATCH_SIZE+1}）: {e}")

    # 去重：同一 term 只保留第一次出现
    for stance in ("fan", "hater", "neutral"):
        seen = set()
        deduped = []
        for item in merged[stance]:
            term = str(item.get("term", "")).strip()
            if term and term not in seen:
                seen.add(term)
                deduped.append(item)
        merged[stance] = deduped

    return merged


# ══════════════════════════════════════════════
#  存储
# ══════════════════════════════════════════════
def _store_slang(subject: str, slang_data: dict, comment_count: int,
                 emb_model: EmbeddingModel, conn):
    """删除旧数据后重新写入，向量化每条 term"""
    # 清除旧数据（重新提炼时覆盖）
    conn.execute("DELETE FROM SlangTerms WHERE subject=?", (subject,))

    now = int(time.time())
    inserted = 0

    for stance in ("fan", "hater", "neutral"):
        for item in slang_data.get(stance, []):
            term = str(item.get("term", "")).strip()
            desc = str(item.get("desc", "")).strip()
            if not term:
                continue

            # 向量化：用 "subject 的 stance 用语：term。desc" 作为文本
            vec_text = f"{subject}的球迷用语（{stance}）：{term}。{desc}"
            dense_mat, sparse_list = emb_model.encode_hybrid([vec_text])
            dense  = dense_mat[0].astype(np.float32).tobytes()
            sparse = json.dumps(sparse_list[0], ensure_ascii=False)

            conn.execute("""
                INSERT OR REPLACE INTO SlangTerms
                    (subject, term, description, stance,
                     vector, sparse_embedding, source_comment_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (subject, term, desc, stance, dense, sparse, comment_count, now))
            inserted += 1

    conn.commit()
    return inserted


# ══════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════
def run(emb_model: EmbeddingModel, llm: OpenAI = None):
    """
    检查所有已分类的篮球区 subject，
    对需要更新的 subject 重新提炼黑话并存储。
    """
    _init_slang_table()

    if llm is None:
        llm = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

    conn     = sqlite3.connect(CONFIG["db_name"])
    subjects = _get_subjects(conn)

    if not subjects:
        logger.info("没有已分类的篮球区帖子，跳过黑话提炼")
        conn.close()
        return

    logger.info(f"🏀 检查 {len(subjects)} 个 subject 的黑话更新需求...")

    updated = 0
    for subject in subjects:
        should_update, comment_count = _needs_update(subject, conn)
        if not should_update:
            logger.info(f"  [{subject}] 无需更新（当前{comment_count}条高赞评论）")
            continue

        logger.info(f"  [{subject}] 开始提炼（{comment_count} 条高赞评论）...")
        comments, _ = _get_comments_for_subject(subject, conn)
        if not comments:
            logger.info(f"  [{subject}] 无高赞评论，跳过")
            continue

        slang_data = _extract_slang(subject, comments, llm)
        n = _store_slang(subject, slang_data, comment_count, emb_model, conn)
        logger.info(f"  [{subject}] 写入 {n} 条黑话词汇")
        updated += 1
        time.sleep(0.5)

    conn.close()
    logger.info(f"✅ 黑话提炼完成：{updated} 个 subject 已更新")


def print_stats():
    """打印当前 SlangTerms 表统计"""
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("""
        SELECT subject, stance, COUNT(*) as n
        FROM SlangTerms
        GROUP BY subject, stance
        ORDER BY subject, stance
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("SlangTerms 表为空")
        return

    print("\n── 黑话库统计 ──────────────────────")
    cur_subject = None
    for subject, stance, n in rows:
        if subject != cur_subject:
            print(f"  【{subject}】")
            cur_subject = subject
        print(f"    {stance:<8} : {n} 条")
    print("─────────────────────────────────────")


if __name__ == "__main__":
    from embedder import EmbeddingModel as _EmbModel
    import os
    os.environ["HF_HOME"] = "E:/models/huggingface"
    _emb = _EmbModel(CONFIG["embedding_model"])
    _llm = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])
    run(_emb, _llm)
    print_stats()
