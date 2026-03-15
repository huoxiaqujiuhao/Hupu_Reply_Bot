"""
memory_store.py — 向量记忆存取
负责 Memories / ReplyContext 表的增删改查、去重、衰减
"""
import sqlite3
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from config import CONFIG, get_logger

logger = get_logger("MemoryStore")


# ══════════════════════════════════════════════
#  数据库初始化
# ══════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(CONFIG["db_name"])
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Memories (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_text        TEXT NOT NULL,
            embedding        BLOB NOT NULL,
            category         TEXT NOT NULL,
            rule_type        TEXT NOT NULL,
            weight           REAL    DEFAULT 1.0,
            reinforced_count INTEGER DEFAULT 1,
            evidence_likes   INTEGER DEFAULT 0,
            source_pid       INTEGER,
            created_at       INTEGER,
            updated_at       INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ReplyContext (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            tid                INTEGER NOT NULL,
            content            TEXT    NOT NULL,
            post_title         TEXT,
            post_content       TEXT,
            post_category      TEXT,
            post_total_replies INTEGER,
            created_at         INTEGER,
            UNIQUE(tid, content)
        )
    """)
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════
#  向量序列化
# ══════════════════════════════════════════════
def _to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()

def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# ══════════════════════════════════════════════
#  添加记忆（含去重）
# ══════════════════════════════════════════════
def add_memory(
    rule_text:     str,
    category:      str,
    rule_type:     str,
    evidence_likes: int,
    source_pid:    int | None,
    emb_model:     SentenceTransformer,
) -> str:
    """
    返回 'reinforced'（强化了已有规则）或 'inserted'（插入了新规则）
    """
    vec = emb_model.encode(rule_text, normalize_embeddings=True).astype(np.float32)
    now = int(time.time())

    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()

    cur.execute(
        "SELECT id, embedding, weight, reinforced_count FROM Memories WHERE category=?",
        (category,)
    )
    rows = cur.fetchall()

    best_id, best_sim = None, 0.0
    for row_id, blob, weight, count in rows:
        sim = float(np.dot(vec, _from_blob(blob)))
        if sim > best_sim:
            best_sim, best_id = sim, row_id

    threshold = CONFIG["memory_dedup_threshold"]

    if best_sim >= threshold:
        cur.execute("""
            UPDATE Memories
            SET weight           = weight + ?,
                reinforced_count = reinforced_count + 1,
                updated_at       = ?
            WHERE id = ?
        """, (CONFIG["memory_weight_boost"], now, best_id))
        conn.commit()
        conn.close()
        logger.debug(f"记忆强化 (sim={best_sim:.2f}): {rule_text[:40]}")
        return "reinforced"
    else:
        cur.execute("""
            INSERT INTO Memories
                (rule_text, embedding, category, rule_type,
                 weight, reinforced_count, evidence_likes, source_pid,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, 1.0, 1, ?, ?, ?, ?)
        """, (_to_blob(vec), rule_text, category, rule_type,
              evidence_likes, source_pid, now, now))
        conn.commit()
        conn.close()
        logger.info(f"新记忆 [{category}/{rule_type}]: {rule_text[:50]}")
        return "inserted"


# ══════════════════════════════════════════════
#  检索回复策略记忆
# ══════════════════════════════════════════════
def search(
    query_text: str,
    category:   str,
    emb_model:  SentenceTransformer,
) -> list[dict]:
    """
    先查同类别，再查 global，按相似度×weight 排序
    返回字段：rule_text, weight, reinforced_count, score
    """
    query_vec = emb_model.encode(query_text, normalize_embeddings=True).astype(np.float32)

    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()

    results = []
    queries = [(category, CONFIG["memory_top_k_category"])]
    if category != "global":
        queries.append(("global", CONFIG["memory_top_k_global"]))

    for cat, top_k in queries:
        cur.execute(
            "SELECT rule_text, embedding, weight, reinforced_count "
            "FROM Memories WHERE category=? AND rule_type != 'negative_dead'",
            (cat,)
        )
        rows = cur.fetchall()
        if not rows:
            continue
        scored = []
        for rule_text, blob, weight, count in rows:
            sim   = float(np.dot(query_vec, _from_blob(blob)))
            score = sim * weight
            scored.append({
                "rule_text":       rule_text,
                "weight":          weight,
                "reinforced_count": count,
                "score":           score,
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        results.extend(scored[:top_k])

    conn.close()
    return results


def search_filter_rules() -> list[str]:
    """返回筛帖规则（filter 类别），按 weight 降序"""
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute(
        "SELECT rule_text FROM Memories WHERE category='filter' "
        "ORDER BY weight DESC LIMIT ?",
        (CONFIG["memory_top_k_filter"],)
    )
    rules = [row[0] for row in cur.fetchall()]
    conn.close()
    return rules


# ══════════════════════════════════════════════
#  记忆衰减（每7天调一次）
# ══════════════════════════════════════════════
_DECAY_FLAG = "data/.last_decay"

def decay_if_needed():
    now = int(time.time())
    try:
        with open(_DECAY_FLAG) as f:
            last = int(f.read().strip())
    except Exception:
        last = 0

    if now - last < 7 * 86400:
        return

    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("""
        UPDATE Memories
        SET weight = weight * ?, updated_at = ?
        WHERE reinforced_count <= ? AND updated_at < ?
    """, (CONFIG["memory_decay_factor"], now,
          CONFIG["memory_decay_protect_count"], now - 7 * 86400))
    cur.execute(
        "DELETE FROM Memories WHERE weight < ?",
        (CONFIG["memory_decay_min_weight"],)
    )
    conn.commit()
    conn.close()
    logger.info("记忆衰减完成")

    with open(_DECAY_FLAG, "w") as f:
        f.write(str(now))


# ══════════════════════════════════════════════
#  ReplyContext 读写（reply_bot.py 发帖后调用）
# ══════════════════════════════════════════════
def save_reply_context(tid: int, content: str, post_title: str,
                       post_content: str, post_category: str,
                       post_total_replies: int):
    conn = sqlite3.connect(CONFIG["db_name"])
    conn.execute("""
        INSERT OR IGNORE INTO ReplyContext
            (tid, content, post_title, post_content,
             post_category, post_total_replies, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (tid, content, post_title,
          (post_content or "")[:CONFIG["memory_content_max_chars"]],
          post_category, post_total_replies, int(time.time())))
    conn.commit()
    conn.close()


def get_reply_context(tid: int, content: str) -> dict | None:
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("""
        SELECT post_title, post_content, post_category, post_total_replies
        FROM ReplyContext WHERE tid=? AND content=?
    """, (tid, content))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "post_title":          row[0],
        "post_content":        row[1],
        "post_category":       row[2],
        "post_total_replies":  row[3],
    }


# ══════════════════════════════════════════════
#  统计摘要
# ══════════════════════════════════════════════
def print_stats():
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("SELECT category, rule_type, COUNT(*), AVG(weight) FROM Memories GROUP BY category, rule_type")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        logger.info("Memories 表为空")
        return
    logger.info("── Memories 摘要 ──────────────────────")
    for cat, rtype, cnt, avg_w in rows:
        logger.info(f"  [{cat}/{rtype}] {cnt} 条，平均权重 {avg_w:.2f}")
    logger.info("───────────────────────────────────────")
