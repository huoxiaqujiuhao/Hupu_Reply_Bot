import sqlite3
import json
import logging
import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
import umap

# ══════════════════════════════════════════════
#  配置区
# ══════════════════════════════════════════════
CONFIG = {
    "db_name":           "hupu_arsenal.db",
    "taxonomy_output":   "taxonomy.json",        # 第一段输出的字典文件
    "embedding_model":   "BAAI/bge-small-zh-v1.5",
    "llm_model":         "qwen-plus",
    "api_key":           "sk-323df3c0e569472a84373f69a7e394a2",
    "base_url":          "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",

    # 聚类参数
    "umap_n_neighbors":  15,
    "umap_n_components": 5,
    "hdbscan_min_cluster_size": 10,   # 至少10帖才算一个类
    "hdbscan_min_samples": 3,
    "centroid_samples":  5,            # 每个簇取距质心最近的 N 个帖子给 LLM 看

    # 收敛参数
    "target_categories": 20,           # 最终期望的分类总数
}

# ══════════════════════════════════════════════
#  日志系统
# ══════════════════════════════════════════════
def init_logger() -> logging.Logger:
    logger = logging.getLogger("TaxonomyGenerator")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler("taxonomy_generator.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ══════════════════════════════════════════════
#  Step 1: 从数据库提取数据
# ══════════════════════════════════════════════
def get_data_from_db(logger) -> pd.DataFrame:
    logger.info("📥 Step 1: 从数据库提取数据...")
    conn = sqlite3.connect(CONFIG["db_name"])

    # 【修复】用窗口函数正确地按帖子分组、按亮数排序，取每个帖子前3条高赞评论
    query = """
    SELECT
        p.url,
        p.title,
        p.title || '。' || COALESCE(SUBSTR(p.content, 1, 200), '') || '。评论：' ||
        COALESCE(top_comments.comments_text, '暂无评论') AS combined_text
    FROM Posts p
    LEFT JOIN (
        SELECT
            post_url,
            GROUP_CONCAT(content, ' | ') AS comments_text
        FROM (
            SELECT post_url, content,
                   ROW_NUMBER() OVER (PARTITION BY post_url ORDER BY lights DESC) AS rn
            FROM Comments
        )
        WHERE rn <= 3
        GROUP BY post_url
    ) top_comments ON p.url = top_comments.post_url
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # 清理空值，防止 Embedding 时崩溃
    df['combined_text'] = df['combined_text'].fillna(df['title'])
    logger.info(f"✅ 成功提取 {len(df)} 篇帖子（含高赞评论）。")
    return df


# ══════════════════════════════════════════════
#  Step 2: Embedding + UMAP + HDBSCAN 聚类
# ══════════════════════════════════════════════
def cluster_texts(df, logger) -> tuple[pd.DataFrame, np.ndarray]:
    logger.info("🧠 Step 2: 向量化 → 降维 → 聚类...")

    # 2-1. Embedding
    logger.info(f"  加载模型 [{CONFIG['embedding_model']}]（首次运行会自动下载）...")
    model = SentenceTransformer(CONFIG["embedding_model"])
    embeddings = model.encode(
        df['combined_text'].tolist(),
        show_progress_bar=True,
        batch_size=64
    )
    logger.info(f"  向量维度: {embeddings.shape}")

    # 2-2. UMAP 降维
    logger.info(f"  UMAP 降维: {embeddings.shape[1]}维 → {CONFIG['umap_n_components']}维...")
    reduced = umap.UMAP(
        n_neighbors=CONFIG["umap_n_neighbors"],
        n_components=CONFIG["umap_n_components"],
        metric="cosine",
        random_state=42
    ).fit_transform(embeddings)

    # 2-3. HDBSCAN 聚类
    clusterer = HDBSCAN(
        min_cluster_size=CONFIG["hdbscan_min_cluster_size"],
        min_samples=CONFIG["hdbscan_min_samples"],
        metric="euclidean"
    )
    df['cluster'] = clusterer.fit_predict(reduced)

    valid_clusters = [c for c in df['cluster'].unique() if c != -1]
    noise_count = (df['cluster'] == -1).sum()
    logger.info(f"✅ 聚类完成：发现 {len(valid_clusters)} 个自然簇，噪音点 {noise_count} 个。")

    return df, reduced, embeddings


# ══════════════════════════════════════════════
#  Step 3: 处理噪音点 —— 最近邻归入已有簇
# ══════════════════════════════════════════════
def reassign_noise(df, embeddings, logger) -> pd.DataFrame:
    noise_mask = df['cluster'] == -1
    noise_count = noise_mask.sum()

    if noise_count == 0:
        logger.info("👍 没有噪音点，跳过归并步骤。")
        return df

    logger.info(f"🔧 Step 3: 用最近邻将 {noise_count} 个噪音点归入已有簇...")

    # 用有效簇的点训练 KNN
    valid_mask = ~noise_mask
    knn = NearestNeighbors(n_neighbors=1, metric="cosine")
    knn.fit(embeddings[valid_mask])

    # 找到每个噪音点最近的有效点，继承它的簇编号
    noise_indices = np.where(noise_mask)[0]
    _, neighbor_indices = knn.kneighbors(embeddings[noise_indices])

    valid_indices = np.where(valid_mask)[0]
    for i, noise_idx in enumerate(noise_indices):
        nearest_valid = valid_indices[neighbor_indices[i][0]]
        df.at[noise_idx, 'cluster'] = df.at[nearest_valid, 'cluster']

    logger.info(f"✅ 所有噪音点已归并，当前无孤立帖子。")
    return df


# ══════════════════════════════════════════════
#  Step 4: 质心抽样 —— 每簇取最具代表性的 N 个
# ══════════════════════════════════════════════
def get_centroid_samples(df, embeddings, cluster_id, n=5) -> list[str]:
    """【修复】取距离质心最近的 N 个帖子，而非随机抽样"""
    mask = df['cluster'] == cluster_id
    cluster_indices = np.where(mask)[0]
    cluster_embeddings = embeddings[cluster_indices]

    # 计算质心
    centroid = cluster_embeddings.mean(axis=0, keepdims=True)

    # 计算每个点到质心的余弦距离
    norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
    centroid_norm = np.linalg.norm(centroid)
    cosine_sims = (cluster_embeddings @ centroid.T) / (norms * centroid_norm + 1e-8)
    cosine_sims = cosine_sims.flatten()

    # 取相似度最高的 N 个
    top_n = min(n, len(cluster_indices))
    top_local_indices = np.argsort(cosine_sims)[::-1][:top_n]
    top_global_indices = cluster_indices[top_local_indices]

    return df.iloc[top_global_indices]['combined_text'].tolist()


# ══════════════════════════════════════════════
#  Step 5: LLM 为每个簇命名
# ══════════════════════════════════════════════
def name_cluster_with_llm(client, samples, cluster_id, cluster_size, logger) -> dict | None:
    system_prompt = """
你是一个虎扑论坛的数据架构师，精通网络黑话。
我会给你一批被数学算法归为同一类的帖子（包含标题、正文片段和高赞评论）。
找出它们的核心共同点，为这个类目命名。

输出必须是严格的 JSON，格式如下：
{
  "reasoning": "简要分析这批帖子的共同主题",
  "primary_category": "2-4个字的一级大类，如：情感相亲、体育竞技、数码科技",
  "secondary_category": "2-4个字的具体二级标签，如：彩礼讨论、NBA赛事、手机选购"
}
"""
    prompt = f"以下是聚类 #{cluster_id}（共 {cluster_size} 个帖子）中最具代表性的 {len(samples)} 个帖子：\n\n"
    for i, text in enumerate(samples):
        prompt += f"【帖子 {i+1}】{text[:300]}\n\n"

    try:
        response = client.chat.completions.create(
            model=CONFIG["llm_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=20
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"  簇 {cluster_id} 命名失败: {e}")
        return None


# ══════════════════════════════════════════════
#  Step 6: LLM 合并相似类，收敛到目标数量
# ══════════════════════════════════════════════
def converge_taxonomy(client, raw_taxonomy, target, logger) -> dict:
    current_count = sum(len(v) for v in raw_taxonomy.values())
    logger.info(f"🔀 Step 6: 合并收敛（当前 {current_count} 个二级类 → 目标 {target} 个）...")

    if current_count <= target:
        logger.info("  当前类数已在目标范围内，跳过合并。")
        return raw_taxonomy

    system_prompt = f"""
你是一个数据架构师。我给你一份从数据中自动生成的分类字典（一级类 → 二级标签列表）。
你的任务是把它合并精简，最终输出的【二级标签总数不超过 {target} 个】。

合并原则：
1. 语义高度相似的二级标签合并成一个，取最有代表性的名字。
2. 如果一级类下只剩1个二级标签，可以把该一级类合并进更大的相邻类。
3. 保持虎扑语境，不要造出过于书面或陌生的词。

输出必须是严格的 JSON，格式与输入完全相同：
{{
  "一级类A": ["二级标签1", "二级标签2"],
  "一级类B": ["二级标签3"]
}}
"""
    prompt = f"请合并以下分类字典，控制二级标签总数在 {target} 个以内：\n\n{json.dumps(raw_taxonomy, ensure_ascii=False, indent=2)}"

    try:
        response = client.chat.completions.create(
            model=CONFIG["llm_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=30
        )
        merged = json.loads(response.choices[0].message.content)
        merged_count = sum(len(v) for v in merged.values())
        logger.info(f"✅ 收敛完成：{current_count} 个二级类 → {merged_count} 个。")
        return merged
    except Exception as e:
        logger.error(f"  收敛失败，使用原始字典: {e}")
        return raw_taxonomy


# ══════════════════════════════════════════════
#  Step 7: 保存结果
# ══════════════════════════════════════════════
def save_taxonomy(taxonomy, df, logger):
    """【修复】把 taxonomy 和簇-帖子对应关系都持久化"""

    # 7-1. 保存 taxonomy.json 给 ai_labeler.py 使用
    with open(CONFIG["taxonomy_output"], "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Taxonomy 字典已保存至: {CONFIG['taxonomy_output']}")

    # 7-2. 把每个帖子的初步聚类结果写回数据库（方便人工复查）
    conn = sqlite3.connect(CONFIG["db_name"])
    try:
        conn.execute("ALTER TABLE Posts ADD COLUMN cluster_id INTEGER")
    except sqlite3.OperationalError:
        pass  # 字段已存在
    conn.commit()

    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("UPDATE Posts SET cluster_id=? WHERE url=?", (int(row['cluster']), row['url']))
    conn.commit()
    conn.close()
    logger.info("💾 各帖子的聚类编号已写回数据库 cluster_id 字段。")


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════
def main():
    logger = init_logger()
    logger.info("🌟 Two-Pass Taxonomy Generator 启动")
    logger.info("=" * 55)

    # Step 1: 取数据
    df = get_data_from_db(logger)

    # Step 2: 聚类
    df, reduced_embeddings, raw_embeddings = cluster_texts(df, logger)

    # Step 3: 处理噪音点
    df = reassign_noise(df, raw_embeddings, logger)

    # Step 4 & 5: 质心抽样 + LLM 命名
    logger.info("\n📝 Step 4&5: 质心抽样 + LLM 为每个簇命名...")
    client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

    raw_taxonomy = {}
    cluster_name_map = {}  # cluster_id → "一级-二级" 的映射，供后续打标使用
    valid_clusters = sorted([c for c in df['cluster'].unique() if c != -1])

    for c in valid_clusters:
        cluster_size = (df['cluster'] == c).sum()
        samples = get_centroid_samples(df, raw_embeddings, c, n=CONFIG["centroid_samples"])
        result = name_cluster_with_llm(client, samples, c, cluster_size, logger)

        if result:
            primary = result.get("primary_category", "未知大类")
            secondary = result.get("secondary_category", "未知子类")

            if primary not in raw_taxonomy:
                raw_taxonomy[primary] = []
            if secondary not in raw_taxonomy[primary]:
                raw_taxonomy[primary].append(secondary)

            cluster_name_map[c] = f"{primary}-{secondary}"
            logger.info(f"  簇 {c:2d} ({cluster_size:3d}帖) → 【{primary}】-【{secondary}】")
            logger.info(f"         推理: {result.get('reasoning', '')[:60]}")

    # Step 6: 收敛到目标类数
    final_taxonomy = converge_taxonomy(client, raw_taxonomy, CONFIG["target_categories"], logger)

    # Step 7: 持久化
    save_taxonomy(final_taxonomy, df, logger)

    # 最终打印
    logger.info("\n" + "=" * 55)
    logger.info("🎉 Taxonomy 生成完毕！最终分类字典：")
    total = sum(len(v) for v in final_taxonomy.values())
    for primary, secondaries in final_taxonomy.items():
        logger.info(f"  📂 {primary}")
        for s in secondaries:
            logger.info(f"       └─ {s}")
    logger.info(f"\n  共 {len(final_taxonomy)} 个一级类，{total} 个二级标签。")
    logger.info(f"  下一步：把 {CONFIG['taxonomy_output']} 载入 ai_labeler.py 开始打标。")


if __name__ == "__main__":
    main()