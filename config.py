"""
config.py — 所有参数和 API Key 都在这里，其他文件不需要改
"""
import logging
import os

os.makedirs("data/logs", exist_ok=True)

# ═══════════════════════════════════════════════════════════
#  ⭐ 在这里调整所有参数
# ═══════════════════════════════════════════════════════════
CONFIG = {

    # ─────────────────────────────────────────────────────
    #  🔑 API（只填一次）
    # ─────────────────────────────────────────────────────
    "api_key":     "sk-a2461abeef1c4ade9642799ab7ca2eb4",
    "base_url":    "https://api.deepseek.com",
    "model":       "deepseek-chat",
    "llm_timeout": 60,

    # ─────────────────────────────────────────────────────
    #  ⏰ 时间控制
    # ─────────────────────────────────────────────────────
    "total_max_hours": 7,     # 程序总寿命（小时）
    "scrape_minutes":  30,    # 每轮爬虫时长（分钟）
    "reply_minutes":   40,    # 每轮回复时长（分钟）

    # 急行军触发：累计回复时间 / 总回复预算 超过此比例 → 急行军
    # 例：0.75 = 前75%从容，最后25%急行军
    # 在 2h、每轮 20min 回复的情况下：总预算≈40min，前30min从容，最后10min急行军
    "panic_ratio": 0.85,

    # ─────────────────────────────────────────────────────
    #  🕷️ 爬虫参数
    # ─────────────────────────────────────────────────────
    "harvest_min_replies":    45,      # 回复数低于此值的帖子不收录
    "post_max_age_hours":     1024,    # 只收这么多小时内的帖子
    "scraper_top_lights":     10,      # 每帖最多收取前 N 条高亮评论
    "allow_image_posts":      True,
    "scraper_delay_posts":    (4, 9),
    "scraper_delay_pages":    (5, 10),
    "scraper_delay_on_error": (15, 30),
    "max_consecutive_errors": 5,
    "long_sleep_on_errors":   (60, 120),
    "max_page_retries":       3,

    # ─────────────────────────────────────────────────────
    #  🏷️ 分类参数
    # ─────────────────────────────────────────────────────
    "embedding_model":    "BAAI/bge-small-zh-v1.5",
    "post_text_max_chars": 400,   # 正文截断长度（用于向量化）
    "vector_batch_size":   64,

    # 余弦相似度低于此值 → other（0~1，越低越宽松；建议先跑几轮再调）
    "other_threshold":    0.40,

    # other 攒够多少条 → 触发重聚类
    "other_recluster_at": 100,

    # 重聚类参数
    "umap_n_neighbors":          15,
    "umap_n_components":         5,
    "hdbscan_min_cluster_size":  8,
    "hdbscan_min_samples":       3,
    "centroid_samples":          5,   # 命名时给 LLM 看几个代表帖子

    # ─────────────────────────────────────────────────────
    #  💬 回复参数
    # ─────────────────────────────────────────────────────
    "max_reply_actions":   10,    # 每轮最多发几条回复

    # 从容模式：评论数在 [min, max] 之间 且 发帖时间在 max_minutes 以内
    "calm_min_replies":    3,
    "calm_max_replies":    10,
    "calm_max_minutes":    60,
    "calm_quality_min":    6,    # AI 质检分数门槛（1-10）

    # 急行军模式
    "panic_golden_max":    30,   # 黄金时段：发帖在此分钟以内
    "panic_scavenge_max":  300,  # 捡漏时段上限（分钟）
    "panic_reply_cap":     25,   # 捡漏时段评论数上限

    # 死帖判定
    "dead_post_max_minutes": 300,
    "dead_post_min_minutes": 60,
    "dead_post_min_replies": 2,
    "too_fresh_minutes":     5,

    # 列表页
    "list_pages":          3,
    "list_url_base":       "https://bbs.hupu.com/topic-daily-postdate",
    "calm_rescan_interval": 120,  # 从容模式没找到合适帖时等待秒数

    # 节奏控制（防封）
    "reply_delay_posts":   (4, 10),
    "reply_delay_pages":   (3, 6),
    "typing_delay_ms":     180,
    "scroll_y_range":      (300, 1200),
    "scroll_pause_1":      (0.5, 1.5),
    "scroll_pause_2":      (0.3, 0.8),
    "pre_click_pause":     (0.5, 1.0),
    "post_click_pause":    (0.3, 0.6),
    "post_type_pause":     (0.8, 1.5),
    "post_send_pause":     (2.0, 3.5),

    # RAG 参数
    "top_k_posts":            5,
    "top_k_comments":         2,
    "min_lights":             10,
    "comment_ref_max_chars":  80,
    "post_prompt_max_chars":  400,
    "current_replies_count":  6,
    "reply_top_lights":       10,

    # LLM 温度
    "classify_temperature":   0.1,
    "analyze_temperature":    0.1,
    "generate_temperature":   0.75,
    "max_tokens":             150,

    # ─────────────────────────────────────────────────────
    #  🧠 反思系统参数
    # ─────────────────────────────────────────────────────
    "memory_positive_threshold":  50,   # light_count >= 此值 → 正样本
    "memory_negative_threshold":  10,   # light_count <= 此值 → 负样本
    "reflect_cooldown_hours":     24,   # 发出不足此小时的评论不反思

    "memory_dedup_threshold":     0.85, # 余弦相似度高于此值 → 强化而非新增
    "memory_weight_boost":        0.3,  # 每次强化增加的权重
    "memory_decay_factor":        0.9,  # 每7天冷记忆权重衰减系数
    "memory_decay_min_weight":    0.1,  # 低于此权重 → 删除
    "memory_decay_protect_count": 5,    # reinforced_count > 此值 → 不衰减

    "memory_top_k_category":      2,    # 检索同类别记忆数量
    "memory_top_k_global":        1,    # 检索 global 记忆数量
    "memory_top_k_filter":        3,    # 注入筛帖 Prompt 的规则数量
    "memory_content_max_chars":   200,  # ReplyContext 存储的正文截断长度

    # ─────────────────────────────────────────────────────
    #  📁 路径（一般不用改）
    # ─────────────────────────────────────────────────────
    "db_name":              "data/hupu_arsenal.db",
    "taxonomy_file":        "data/taxonomy.json",
    "centroids_file":       "data/centroids.json",
    "replied_history_file": "data/replied_urls.json",
    "log_file":             "data/logs/hupu.log",
}


# ═══════════════════════════════════════════════════════════
#  日志系统（所有模块共用同一个日志文件 + 终端输出）
# ═══════════════════════════════════════════════════════════
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] [%(name)-11s] %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(CONFIG["log_file"], encoding="utf-8")
    fh.setLevel(logging.DEBUG)   # 文件记录全部
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)    # 终端只显示 INFO 以上
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger