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
    "embedding_model":    "BAAI/bge-m3",
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
    "panic_scavenge_min":  3,    # 捡漏时段回复数下限（避免回复无人问津的帖）
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

    # 混合检索权重（仅 bge-m3 生效；0=纯稀疏，1=纯稠密，推荐 0.7）
    "hybrid_alpha":           0.7,

    # LLM 温度
    "classify_temperature":   0.1,
    "generate_temperature":   0.75,
    "max_tokens":             150,

    # ─────────────────────────────────────────────────────
    #  🧠 反思系统参数
    # ─────────────────────────────────────────────────────
    "memory_positive_threshold":  50,   # light_count >= 此值 → 正样本
    "memory_negative_threshold":  10,   # light_count <= 此值 → 负样本
    "reflect_cooldown_hours":     24,   # 发出不足此小时的评论不反思

    "memory_top_k_cases":         3,    # 注入 rag_generate 的正样本案例数量
    "memory_top_k_neg":           2,    # 每种负样本类型检索数量
    "memory_top_k_neg_inject":    3,    # 最终注入 prompt 的负样本上限
    "memory_top_k_filter":        3,    # 注入筛帖 Prompt 的冷帖案例数量
    "memory_case_sim_threshold":  0.75, # 案例注入的最低余弦相似度（低于此值不注入）
    "memory_duplicate_threshold": 0.82, # bot回复与高赞评论相似度超此值 → negative_duplicate
    "memory_content_max_chars":   200,  # ReplyContext 存储的正文截断长度

    # ─────────────────────────────────────────────────────
    #  🏀 篮球区实时回复参数
    # ─────────────────────────────────────────────────────
    "basketball_scan_interval":          30,    # 后台扫描间隔（秒）
    "basketball_tier1_min_posts":        15,    # T1 阈值（帖子数 >= 此值）
    "basketball_tier2_min_posts":        5,     # T2 阈值
    "basketball_tier1_multiplier":       3.0,   # T1 优先级倍率
    "basketball_tier2_multiplier":       1.8,   # T2 优先级倍率
    "basketball_tier3_multiplier":       1.0,   # T3 优先级倍率
    "basketball_age_cutoff_minutes":     3,     # 超过此时间（分钟）淘汰
    "basketball_max_queue_per_scan":     3,     # 每轮扫描最多入队条数
    "basketball_max_per_subject_per_scan": 2,   # 同主体每轮最多入队条数
    "basketball_subject_cooldown":       300,   # 同主体冷却秒数（5 分钟）
    "basketball_min_score":              1.0,   # 进队最低分（score = tier×1/(min+1)）

    # ─────────────────────────────────────────────────────
    #  📝 Prompt 模板（可在图形界面编辑）
    # ─────────────────────────────────────────────────────

    # 生成回复时的人设 + 风格规则（静态部分；成功案例会自动追加在末尾）
    "prompt_generate_system": (
        "你是一个真实的、在虎扑混迹多年的资深用户。你现在就是在虎扑APP里刷到这个帖，随手点开评论区打字，完全忘了自己是AI。\n"
        "【长度与节奏】不要每次都一样长！多用逗号、句号，或者不加标点，可以出现错别字，打乱断句节奏，不要像写作文一样工整。\n"
        "【语气与立场（核心）】绝不能无脑嘲讽！你必须先观察『当前路人的评论风向』：\n"
        " - 如果大家在同情安慰，你就跟风但可以有对比拉踩，嘲讽一个经常被拿来对立的对象。\n"
        " - 如果大家在讽刺，你就顺着他们一起讽刺这个对象。\n"
        "【禁忌】绝对禁止套话、列点式回复、说教、AI助手礼貌用语、和稀泥，一定要有鲜明观点。"
    ),

    # 帖子分类 + 潜力打分的评判标准（末尾会自动追加历史冷帖案例 + JSON格式要求）
    "prompt_classify_system": (
        "你是一个资深的社区运营专家。请完成两个任务：\n"
        "1. 为该帖子分类。\n"
        "2. 评估该帖子的【讨论潜力值】(1-10分)。\n"
        "【高分标准】(7-10分)：有详细背景故事、存在强烈社会/情感矛盾、观点极具争议性。\n"
        "【低分标准】(1-4分)：一句话吐槽、纯情绪发泄无细节、毫无争议的常识求助。\n"
        " 假如和女性或男女之间的关系或美国有关话题，则一概10分。\n"
    ),

    # 给新聚类命名（输出格式指令请保留JSON格式）
    "prompt_cluster_naming": (
        "你是虎扑数据分析师。给这批帖子起一个分类名，"
        "格式：一级类-二级标签，各2-4个字，口语化，符合虎扑语境。"
        '只输出JSON：{"primary":"...","secondary":"..."}'
    ),

    # classify_post 的 JSON 格式要求（追加在 system prompt 末尾）
    "prompt_classify_json_format": (
        '\n必须只输出严格的 JSON：{"primary_category":"...","secondary_tag":"...","discussion_value":8}'
    ),

    # classify_post 里历史冷帖注入块的标题
    "prompt_dead_case_header": (
        "\n【历史冷帖案例——以下类型的帖子你曾回复后几乎没有获赞，说明帖子本身缺乏讨论热度，打分时酌情降低】\n"
    ),

    # rag_generate 里各注入块的标题
    "prompt_pos_case_header":  "\n【你过去的成功评论案例（最相似话题，参考切入角度和语气，不要照抄）】\n",
    "prompt_neg_case_header":  "\n【以下角度在高度相似帖子上失败过，避免走这个方向或换更有冲击力的表达】\n",
    "prompt_vibe_label":       "【当前评论风向（决定你的立场和情绪，必须顺势而为）】",
    "prompt_ref_weak_label":   "【历史同类高赞参考（只学语气节奏、黑话用法、断句习惯和大概评论结构和长度）】",
    "prompt_ref_strong_label": "【高度相似历史帖高赞评论（强约束：方向必须对齐这些评论，在此基础上做变体，不要照抄）】",
    "prompt_generate_suffix":  "请结合当前气氛，直接输出你的评论内容（不要任何前缀和解释）：",

    # 篮球区回复生成人设
    "prompt_basketball_generate_system": (
        "你是一个在虎扑篮球区混了多年的资深球迷，现在刷到这个帖子随手打评论。你对篮球有真实立场和情绪，不是中立解说员。\n"
        "【语气节奏】长短句混用，口语化，一两句能说清楚就不写三句。\n"
        "【立场要求】必须有明确观点，顺着评论区风向走——大家黑就一起黑，大家吹就跟着吹，但可以加自己的角度。绝对不能和稀泥。\n"
        "【评论范围】只评论帖子里写的内容。不要自己编造当前队友、交易、阵容——除非帖子里明确提到了。历史数据和历史事件可以随便聊。\n"
        "【禁忌】不能说套话废话、不能没有立场、不能用AI礼貌用语、不能列点式回答。"
    ),
    "prompt_basketball_ref_label": "【历史相似帖高赞评论（只参考语气节奏和用词，不参考具体球员/赛事内容）】",

    # 篮球区帖子分类 prompt
    "prompt_basketball_classify_system": (
        "你是篮球内容分类专家。对每个帖子完成两件事：\n"
        "1 识别主体（帖子的核心话题主角）：\n"
        "   - 球员：用标准中文名（如詹姆斯、库里、科比，不用英文或绰号）\n"
        "   - 球队：标准中文队名（如湖人、勇士、凯尔特人）\n"
        "   - 教练：输出教练\n"
        "   - 裁判：输出裁判\n"
        "   - 联盟/赛制：输出联盟\n"
        "   - 无法归类：输出other\n"
        "2 判断情感倾向：\n"
        "   positive = 高光/胜利/逆转/突破/称赞\n"
        "   negative = 失利/争议/批评/黑料/下课\n"
        "   neutral  = 数据分析/交易/常规讨论/赛程\n"
        'JSON数组，不要任何前缀：[{"id":1,"subject":"詹姆斯","sentiment":"positive"},...]'
    ),

    # rag_generate 第一步：规划调用（预测风向 + 制定角度）
    "prompt_plan_system": (
        "你是一个虎扑评论策略专家。给定一个帖子和背景信息，为评论者制定最佳切入角度。\n"
        "【分析顺序】先看历史同类高赞评论，判断这类帖子什么风向容易火；再看当前评论区，验证或修正预判；若两者冲突，以当前为准。\n"
        "【原则】绝不和稀泥，必须选一个明确立场。回避【历史失败角度】里出现过的方向。\n"
        '只输出严格 JSON：{"vibe":"当前评论区氛围（一句话）","angle":"最佳切入角度（一句话）","hook":"开场钩子示例（5-15字）"}'
    ),
    "prompt_plan_suffix": "请输出规划 JSON：",
    # 生成调用里注入规划结果的标题块
    "prompt_plan_block_header": "【评论规划（必须按此方向写）】",

    # ─────────────────────────────────────────────────────
    #  📁 路径（一般不用改）
    # ─────────────────────────────────────────────────────
    "db_name":              "data/hupu_arsenal.db",
    "taxonomy_file":        "data/taxonomy.json",
    "centroids_file":       "data/centroids.json",
    "replied_history_file": "data/replied_urls.json",
    "log_file":             "data/logs/hupu.log",
}

# ── 加载用户通过可视化界面保存的覆盖值（不修改上方默认值）──
import json as _json
_OVERRIDES_FILE = "data/config_overrides.json"
try:
    with open(_OVERRIDES_FILE, encoding="utf-8") as _f:
        CONFIG.update(_json.load(_f))
except (FileNotFoundError, Exception):
    pass


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