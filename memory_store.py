"""
memory_store.py — 案例存取
Cases 表：存储 Bot 历史发言的原始案例（帖子+回复+结果）
检索时用混合向量相似度（稠密 + 稀疏），找出最相关的成功/失败案例注入 Prompt
"""
import sqlite3
import json
import time
import numpy as np
from config import CONFIG, get_logger
from embedder import EmbeddingModel, sparse_dot

logger = get_logger("MemoryStore")


# ══════════════════════════════════════════════
#  数据库初始化
# ══════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(CONFIG["db_name"])
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS Cases (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            tid              INTEGER UNIQUE,
            post_title       TEXT,
            post_content     TEXT,
            bot_reply        TEXT,
            light_count      INTEGER DEFAULT 0,
            category         TEXT,
            case_type        TEXT,
            embedding        BLOB,
            sparse_embedding TEXT,
            created_at       INTEGER
        )
    """)
    # 迁移：给已存在但没有 sparse_embedding 列的旧表加列
    try:
        conn.execute("ALTER TABLE Cases ADD COLUMN sparse_embedding TEXT")
    except Exception:
        pass  # 已存在则忽略

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

def _sparse_to_json(sparse: dict) -> str:
    return json.dumps(sparse, ensure_ascii=False)

def _sparse_from_json(s: str | None) -> dict:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


# ══════════════════════════════════════════════
#  混合得分
# ══════════════════════════════════════════════
def _hybrid_sim(q_dense: np.ndarray, d_dense: np.ndarray,
                q_sparse: dict, d_sparse: dict) -> float:
    alpha = CONFIG["hybrid_alpha"]
    d_score = float(np.dot(q_dense, d_dense))
    s_score = sparse_dot(q_sparse, d_sparse)
    return alpha * d_score + (1.0 - alpha) * s_score


# ══════════════════════════════════════════════
#  添加案例
# ══════════════════════════════════════════════
def add_case(
    tid:          int,
    post_title:   str,
    post_content: str,
    bot_reply:    str,
    light_count:  int,
    category:     str,
    case_type:    str,
    emb_model:    EmbeddingModel,
) -> bool:
    """
    插入一条案例，tid 唯一。
    返回 True=新插入，False=已存在跳过
    """
    text = f"{post_title}。{(post_content or '')[:CONFIG['post_text_max_chars']]}"
    dense_mat, sparse_list = emb_model.encode_hybrid([text])
    dense  = dense_mat[0]
    sparse = sparse_list[0]

    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO Cases
                (tid, post_title, post_content, bot_reply,
                 light_count, category, case_type,
                 embedding, sparse_embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tid, post_title,
              (post_content or "")[:CONFIG["memory_content_max_chars"]],
              bot_reply, light_count, category, case_type,
              _to_blob(dense), _sparse_to_json(sparse), int(time.time())))
        conn.commit()
        inserted = True
    except sqlite3.IntegrityError:
        inserted = False
    conn.close()
    return inserted


# ══════════════════════════════════════════════
#  检索案例（混合向量相似度）
# ══════════════════════════════════════════════
def search_cases(
    query_text:  str,
    case_type:   str,
    emb_model:   EmbeddingModel,
    category:    str = None,
    top_k:       int = None,
) -> list[dict]:
    """
    按混合相似度检索案例（稠密 + 稀疏，non-M3 自动退化为纯稠密）。
    category 不为 None 时优先同类，不足则补全局。
    """
    if top_k is None:
        top_k = CONFIG["memory_top_k_cases"]

    dense_mat, sparse_list = emb_model.encode_hybrid([query_text])
    q_dense  = dense_mat[0]
    q_sparse = sparse_list[0]

    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute(
        "SELECT tid, post_title, post_content, bot_reply, "
        "light_count, category, embedding, sparse_embedding "
        "FROM Cases WHERE case_type=?",
        (case_type,)
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    scored = []
    for tid, title, content, reply, likes, cat, blob, sparse_json in rows:
        d_dense  = _from_blob(blob)
        d_sparse = _sparse_from_json(sparse_json)
        sim = _hybrid_sim(q_dense, d_dense, q_sparse, d_sparse)
        scored.append({
            "tid":          tid,
            "post_title":   title,
            "post_content": content,
            "bot_reply":    reply,
            "light_count":  likes,
            "category":     cat,
            "similarity":   sim,
        })

    # 同类优先
    if category:
        same  = [x for x in scored if x["category"] == category]
        other = [x for x in scored if x["category"] != category]
        same.sort(key=lambda x: x["similarity"],  reverse=True)
        other.sort(key=lambda x: x["similarity"], reverse=True)
        merged = same[:top_k] + other[:max(0, top_k - len(same[:top_k]))]
        return merged[:top_k]
    else:
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]


# ══════════════════════════════════════════════
#  ReplyContext 读写
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
    cur.execute("SELECT case_type, COUNT(*) FROM Cases GROUP BY case_type")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        logger.info("Cases 表为空")
        return
    logger.info("── Cases 摘要 ────────────────────────")
    for ctype, cnt in rows:
        logger.info(f"  [{ctype}] {cnt} 条")
    logger.info("─────────────────────────────────────")
