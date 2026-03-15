"""
classifier.py — 质心相似度分类 + other 重聚类

主要逻辑：
  1. classify_posts()：对数据库里所有 ai_tag IS NULL 的帖子，
     计算它和每个已知类别质心的余弦相似度。
     最高相似度 >= other_threshold → 打上对应标签
     最高相似度 <  other_threshold → 标为 "other"
     全程不调 LLM。

  2. check_and_recluster_others()：当 other 数量达到阈值时，
     对 other 帖子做 UMAP+HDBSCAN 聚类，LLM 命名新子类，
     更新 taxonomy.json 并立刻重建所有质心。
"""
import sqlite3
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from config import CONFIG, get_logger

logger = get_logger("Classifier")


# ══════════════════════════════════════════════
#  质心文件 I/O
# ══════════════════════════════════════════════
def load_centroids() -> dict[str, np.ndarray]:
    """从磁盘加载缓存质心，返回 {ai_tag: ndarray(dim,)}"""
    try:
        with open(CONFIG["centroids_file"], encoding="utf-8") as f:
            raw = json.load(f)
        return {tag: np.array(vec, dtype=np.float32) for tag, vec in raw.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_centroids(centroids: dict[str, np.ndarray]):
    with open(CONFIG["centroids_file"], "w", encoding="utf-8") as f:
        json.dump({tag: vec.tolist() for tag, vec in centroids.items()}, f)
    logger.debug(f"质心已保存：{len(centroids)} 个类别")


def rebuild_centroids(emb_model: SentenceTransformer) -> dict[str, np.ndarray]:
    """
    从数据库重建所有类别的质心（不含 other 和 ERROR）。
    耗时操作，只在以下情况调用：
      - 首次启动且 centroids.json 不存在
      - other 重聚类完成后
    """
    logger.info("🔄 重建所有质心（从数据库全量计算）...")
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("""
        SELECT ai_tag, title, content FROM Posts
        WHERE ai_tag IS NOT NULL
          AND ai_tag NOT LIKE 'ERROR%'
          AND ai_tag != 'other'
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        logger.warning("数据库中没有已打标帖子，质心为空。")
        return {}

    tag_texts: dict[str, list[str]] = {}
    for ai_tag, title, content in rows:
        text = f"{title}。{(content or '')[:CONFIG['post_text_max_chars']]}"
        tag_texts.setdefault(ai_tag, []).append(text)

    centroids = {}
    for tag, texts in tag_texts.items():
        vecs = emb_model.encode(
            texts,
            batch_size=CONFIG["vector_batch_size"],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[tag] = centroid / norm if norm > 0 else centroid

    save_centroids(centroids)
    logger.info(f"✅ 质心重建完成：{len(centroids)} 个类别，"
                f"共处理 {len(rows)} 个帖子")
    return centroids


# ══════════════════════════════════════════════
#  主分类函数（不调 LLM）
# ══════════════════════════════════════════════
def classify_posts(emb_model: SentenceTransformer) -> int:
    """
    对所有 ai_tag IS NULL 的帖子进行分类。
    返回本次处理的帖子数量。
    """
    centroids = load_centroids()
    if not centroids:
        logger.warning("⚠️ 质心文件为空，跳过分类。"
                       "（请确保数据库已有打标帖子，或先运行 rebuild_centroids）")
        return 0

    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("SELECT url, title, content FROM Posts WHERE ai_tag IS NULL")
    rows = cur.fetchall()

    if not rows:
        logger.info("没有未分类的帖子，跳过。")
        conn.close()
        return 0

    logger.info(f"🏷️  开始分类 {len(rows)} 个未标注帖子 "
                f"（阈值 {CONFIG['other_threshold']}）...")

    # 取出现有各类别帖子数（用于后续增量更新质心）
    tag_counts: dict[str, int] = {}
    for tag in centroids:
        cur.execute("SELECT COUNT(*) FROM Posts WHERE ai_tag=?", (tag,))
        tag_counts[tag] = cur.fetchone()[0]
    conn.close()

    tags_list       = list(centroids.keys())
    centroid_matrix = np.stack([centroids[t] for t in tags_list])   # (n_tags, dim)

    texts = [
        f"{title}。{(content or '')[:CONFIG['post_text_max_chars']]}"
        for _, title, content in rows
    ]
    vecs = emb_model.encode(
        texts,
        batch_size=CONFIG["vector_batch_size"],
        show_progress_bar=False,
        normalize_embeddings=True,
    )  # (n_posts, dim)

    # 余弦相似度（向量已归一化，直接矩阵乘法）
    sims          = vecs @ centroid_matrix.T          # (n_posts, n_tags)
    best_idx      = np.argmax(sims, axis=1)           # (n_posts,)
    best_sim      = sims[np.arange(len(rows)), best_idx]  # (n_posts,)

    threshold = CONFIG["other_threshold"]
    assigned_count = 0
    other_count    = 0
    # {tag: [new_vec, ...]} 用于增量更新质心
    newly_assigned: dict[str, list[np.ndarray]] = {}

    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    for i, (url, title, _) in enumerate(rows):
        sim = float(best_sim[i])
        if sim >= threshold:
            tag = tags_list[best_idx[i]]
            assigned_count += 1
            newly_assigned.setdefault(tag, []).append(vecs[i])
            logger.debug(f"  ✓ {tag}（{sim:.3f}）: {title[:30]}")
        else:
            tag = "other"
            other_count += 1
            logger.debug(f"  ? other（最高 {sim:.3f}）: {title[:30]}")
        cur.execute("UPDATE Posts SET ai_tag=? WHERE url=?", (tag, url))

    conn.commit()
    conn.close()

    logger.info(f"✅ 分类完成：{assigned_count} 已归类 | {other_count} → other")

    # 增量更新被新帖影响的质心（避免全量重算）
    if newly_assigned:
        _update_centroids_incrementally(centroids, tag_counts, newly_assigned)

    return len(rows)


def _update_centroids_incrementally(
    centroids: dict[str, np.ndarray],
    old_counts: dict[str, int],
    newly_assigned: dict[str, list[np.ndarray]],
):
    """加权平均更新质心，只动有新帖进入的类别"""
    for tag, new_vecs in newly_assigned.items():
        old_count  = max(1, old_counts.get(tag, 1))
        new_count  = len(new_vecs)
        new_mat    = np.stack(new_vecs)
        updated    = (centroids[tag] * old_count + new_mat.sum(axis=0)) / (old_count + new_count)
        norm       = np.linalg.norm(updated)
        centroids[tag] = updated / norm if norm > 0 else updated
    save_centroids(centroids)
    logger.debug(f"  质心增量更新：{len(newly_assigned)} 个类别已更新")


# ══════════════════════════════════════════════
#  other 重聚类
# ══════════════════════════════════════════════
def check_and_recluster_others(emb_model: SentenceTransformer, llm: OpenAI) -> bool:
    """
    检查 other 数量，达到阈值则重聚类，生成新子类，重建全量质心。
    返回是否触发了重聚类。
    """
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM Posts WHERE ai_tag='other'")
    other_count = cur.fetchone()[0]
    conn.close()

    if other_count < CONFIG["other_recluster_at"]:
        logger.info(f"  other 当前 {other_count} 个 / 阈值 {CONFIG['other_recluster_at']}，暂不重聚类")
        return False

    logger.info(f"🔀 other 达到 {other_count} 个，触发重聚类...")

    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    cur.execute("SELECT url, title, content FROM Posts WHERE ai_tag='other'")
    other_rows = cur.fetchall()
    conn.close()

    texts = [
        f"{title}。{(content or '')[:CONFIG['post_text_max_chars']]}"
        for _, title, content in other_rows
    ]
    vecs = emb_model.encode(
        texts,
        batch_size=CONFIG["vector_batch_size"],
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # UMAP 降维
    import umap
    from sklearn.cluster import HDBSCAN
    from sklearn.neighbors import NearestNeighbors

    logger.info("  UMAP 降维...")
    reduced = umap.UMAP(
        n_neighbors=CONFIG["umap_n_neighbors"],
        n_components=CONFIG["umap_n_components"],
        metric="cosine",
        random_state=42,
    ).fit_transform(vecs)

    # HDBSCAN 聚类
    logger.info("  HDBSCAN 聚类...")
    clusterer = HDBSCAN(
        min_cluster_size=CONFIG["hdbscan_min_cluster_size"],
        min_samples=CONFIG["hdbscan_min_samples"],
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    # 噪音点（-1）用最近邻归入有效簇
    noise_mask = labels == -1
    if noise_mask.any():
        valid_mask = ~noise_mask
        if valid_mask.sum() > 0:
            knn = NearestNeighbors(n_neighbors=1, metric="cosine")
            knn.fit(vecs[valid_mask])
            _, nbrs = knn.kneighbors(vecs[noise_mask])
            valid_idx = np.where(valid_mask)[0]
            for i, ni in enumerate(np.where(noise_mask)[0]):
                labels[ni] = labels[valid_idx[nbrs[i][0]]]
        else:
            # 全部都是噪音，无法聚类
            logger.warning("  重聚类结果全为噪音，other 保持不变。")
            return False

    valid_clusters = [c for c in np.unique(labels) if c != -1]
    if not valid_clusters:
        logger.warning("  未发现有效聚类，other 保持不变。")
        return False

    logger.info(f"  发现 {len(valid_clusters)} 个新聚类，调用 LLM 命名...")
    new_tags = _name_new_clusters(other_rows, vecs, labels, valid_clusters, llm)

    # 写回数据库
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()
    for i, (url, _, _) in enumerate(other_rows):
        new_tag = new_tags.get(int(labels[i]), "其他-未归类")
        cur.execute("UPDATE Posts SET ai_tag=? WHERE url=?", (new_tag, url))
    conn.commit()
    conn.close()

    # 更新 taxonomy.json
    _update_taxonomy(list(new_tags.values()))

    # 立刻重建全量质心（新类别立即生效）
    rebuild_centroids(emb_model)

    logger.info(f"✅ 重聚类完成，新增 {len(new_tags)} 个子类别："
                f" {', '.join(new_tags.values())}")
    return True


def _name_new_clusters(
    rows: list,
    vecs: np.ndarray,
    labels: np.ndarray,
    valid_clusters: list,
    llm: OpenAI,
) -> dict[int, str]:
    """用 LLM 给每个新聚类命名，返回 {cluster_id: 'primary-secondary'}"""
    n_samples = CONFIG["centroid_samples"]
    new_tags  = {}

    for c in valid_clusters:
        mask       = labels == c
        c_idx      = np.where(mask)[0]
        c_vecs     = vecs[c_idx]
        centroid   = c_vecs.mean(axis=0)
        sims       = c_vecs @ centroid
        top_idx    = np.argsort(sims)[::-1][:n_samples]

        samples_text = ""
        for j, idx in enumerate(top_idx):
            _, title, content = rows[c_idx[idx]]
            samples_text += f"{j+1}. {title}\n"

        try:
            resp = llm.chat.completions.create(
                model=CONFIG["model"],
                messages=[
                    {"role": "system", "content":
                        "你是虎扑数据分析师。给这批帖子起一个分类名，"
                        "格式：一级类-二级标签，各2-4个字，口语化，符合虎扑语境。"
                        '只输出JSON：{"primary":"...","secondary":"..."}'},
                    {"role": "user", "content":
                        f"这批帖子（共 {mask.sum()} 个）代表样本：\n{samples_text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                timeout=CONFIG["llm_timeout"],
            )
            r   = json.loads(resp.choices[0].message.content)
            tag = f"{r['primary']}-{r['secondary']}"
            new_tags[c] = tag
            logger.info(f"  簇 {c}（{mask.sum()} 帖）→ 【{tag}】")
        except Exception as e:
            fallback = f"新类别-{c}"
            new_tags[c] = fallback
            logger.error(f"  簇 {c} 命名失败: {e}，使用默认名 [{fallback}]")

    return new_tags


def _update_taxonomy(new_tag_strings: list[str]):
    """把新类别追加进 taxonomy.json"""
    try:
        with open(CONFIG["taxonomy_file"], encoding="utf-8") as f:
            taxonomy = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        taxonomy = {}

    added = []
    for tag in new_tag_strings:
        parts = tag.split("-", 1)
        if len(parts) == 2:
            primary, secondary = parts
            if primary not in taxonomy:
                taxonomy[primary] = []
            if secondary not in taxonomy[primary]:
                taxonomy[primary].append(secondary)
                added.append(tag)

    with open(CONFIG["taxonomy_file"], "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)
    logger.info(f"  taxonomy.json 已更新，新增：{added}")