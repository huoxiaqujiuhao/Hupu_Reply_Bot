"""
config_editor.py — 参数可视化编辑器
运行后自动打开 Edge 浏览器，所有修改保存至 data/config_overrides.json
"""
import html as _html
import json
import os
import subprocess
import threading
import time
import webbrowser
from flask import Flask, request, jsonify
from config import CONFIG

app = Flask(__name__)
os.makedirs("data", exist_ok=True)
OVERRIDES_FILE = "data/config_overrides.json"

# ══════════════════════════════════════════════
#  参数元数据：分组 + 中文描述 + 类型
# ══════════════════════════════════════════════
PARAM_META = {
    # ─── API 设置 ────────────────────────────
    "api_key":   {"group": "API 设置", "label": "API Key", "desc": "OpenAI 兼容接口的密钥", "type": "password"},
    "base_url":  {"group": "API 设置", "label": "API Base URL", "desc": "接口地址，默认 DeepSeek", "type": "str"},
    "model":     {"group": "API 设置", "label": "模型名称", "desc": "用于生成回复和反思的主力模型", "type": "str"},
    "llm_timeout": {"group": "API 设置", "label": "LLM 超时（秒）", "desc": "单次 API 请求的最长等待时间", "type": "int"},

    # ─── 时间控制 ────────────────────────────
    "total_max_hours":  {"group": "时间控制", "label": "程序总寿命（小时）", "desc": "main.py 单次运行的最长时间，超时自动退出", "type": "float"},
    "scrape_minutes":   {"group": "时间控制", "label": "每轮爬虫时长（分钟）", "desc": "每轮循环中爬虫阶段占用的时间", "type": "int"},
    "reply_minutes":    {"group": "时间控制", "label": "每轮回复时长（分钟）", "desc": "每轮循环中回复阶段占用的时间", "type": "int"},
    "panic_ratio":      {"group": "时间控制", "label": "急行军触发比例", "desc": "累计回复时间/总预算超过此比例时切换急行军模式，0.85=前85%从容", "type": "float"},

    # ─── 爬虫参数 ────────────────────────────
    "harvest_min_replies":    {"group": "爬虫参数", "label": "收录最低回复数", "desc": "帖子回复数低于此值不收入数据库", "type": "int"},
    "post_max_age_hours":     {"group": "爬虫参数", "label": "帖子最大年龄（小时）", "desc": "只收录发出不超过此时间的帖子，设大值=不限", "type": "int"},
    "scraper_top_lights":     {"group": "爬虫参数", "label": "收录高赞评论数", "desc": "每个帖子最多收录前 N 条高赞评论进数据库", "type": "int"},
    "allow_image_posts":      {"group": "爬虫参数", "label": "允许图片帖", "desc": "是否收录含图片的帖子", "type": "bool"},
    "max_consecutive_errors": {"group": "爬虫参数", "label": "最大连续错误数", "desc": "连续出错超过此次数后进入长休眠", "type": "int"},
    "max_page_retries":       {"group": "爬虫参数", "label": "单页最大重试次数", "desc": "单个页面加载失败时的重试次数", "type": "int"},

    # ─── 分类参数 ────────────────────────────
    "embedding_model":         {"group": "分类参数", "label": "Embedding 模型", "desc": "用于向量化的本地模型，首次运行会自动下载", "type": "str"},
    "post_text_max_chars":     {"group": "分类参数", "label": "正文向量化截断长度", "desc": "向量化时正文只取前 N 字，越长越慢", "type": "int"},
    "vector_batch_size":       {"group": "分类参数", "label": "向量批处理大小", "desc": "一次同时向量化多少条文本，显存不足时调小", "type": "int"},
    "other_threshold":         {"group": "分类参数", "label": "other 阈值（0~1）", "desc": "余弦相似度低于此值的帖子归为 other，越低越宽松（建议0.35~0.50）", "type": "float"},
    "other_recluster_at":      {"group": "分类参数", "label": "other 重聚类触发数", "desc": "other 积累到此数量时自动触发重聚类、产生新分类", "type": "int"},
    "umap_n_neighbors":        {"group": "分类参数", "label": "UMAP 邻居数", "desc": "UMAP 降维时的邻居参数，影响聚类粗细，建议 10~20", "type": "int"},
    "umap_n_components":       {"group": "分类参数", "label": "UMAP 降维维度", "desc": "UMAP 输出的维度数，建议 3~8", "type": "int"},
    "hdbscan_min_cluster_size":{"group": "分类参数", "label": "HDBSCAN 最小簇大小", "desc": "低于此数量的聚类被丢弃，越小越容易产生新分类", "type": "int"},
    "hdbscan_min_samples":     {"group": "分类参数", "label": "HDBSCAN 最小样本数", "desc": "控制聚类密度要求，越大越严格", "type": "int"},
    "centroid_samples":        {"group": "分类参数", "label": "质心命名样本数", "desc": "给 LLM 命名新聚类时展示几个代表帖子", "type": "int"},

    # ─── 回复参数 ────────────────────────────
    "max_reply_actions":    {"group": "回复参数", "label": "每轮最多回复数", "desc": "每轮回复阶段最多发送的评论数量", "type": "int"},
    "calm_min_replies":     {"group": "回复参数", "label": "从容模式：最少回复数", "desc": "从容模式下，帖子已有回复数必须 ≥ 此值", "type": "int"},
    "calm_max_replies":     {"group": "回复参数", "label": "从容模式：最多回复数", "desc": "从容模式下，帖子已有回复数必须 ≤ 此值（黄金坑位）", "type": "int"},
    "calm_max_minutes":     {"group": "回复参数", "label": "从容模式：发帖最大分钟", "desc": "从容模式只回复发出不超过此分钟的帖子", "type": "int"},
    "calm_quality_min":     {"group": "回复参数", "label": "从容模式：质检最低分", "desc": "从容模式下 AI 给帖子打的潜力分必须 ≥ 此值（1-10）", "type": "int"},
    "calm_rescan_interval": {"group": "回复参数", "label": "从容模式：无帖等待秒数", "desc": "从容模式没找到合适帖子时等待多少秒再扫描", "type": "int"},
    "panic_golden_max":     {"group": "回复参数", "label": "急行军：黄金时段分钟", "desc": "急行军模式下，发帖在此分钟内属黄金时段，直接回复", "type": "int"},
    "panic_scavenge_max":   {"group": "回复参数", "label": "急行军：捡漏时段上限（分钟）", "desc": "急行军模式下的捡漏时段，超过此分钟不再回复", "type": "int"},
    "panic_reply_cap":      {"group": "回复参数", "label": "急行军：捡漏评论数上限", "desc": "捡漏时段帖子回复数超过此值则跳过（坑位已满）", "type": "int"},
    "dead_post_max_minutes":{"group": "回复参数", "label": "死帖：最大发帖分钟", "desc": "帖子发出超过此分钟直接跳过，无论回复数多少", "type": "int"},
    "dead_post_min_minutes":{"group": "回复参数", "label": "死帖：最小发帖分钟", "desc": "发出超过此分钟且回复数少于死帖最少回复数时跳过", "type": "int"},
    "dead_post_min_replies":{"group": "回复参数", "label": "死帖：最少回复数", "desc": "配合死帖最小发帖分钟使用，超时且回复少于此值则跳过", "type": "int"},
    "too_fresh_minutes":    {"group": "回复参数", "label": "太新：最短等待分钟", "desc": "急行军模式下，帖子发出不足此分钟则等子弹飞", "type": "int"},
    "list_pages":           {"group": "回复参数", "label": "列表页扫描页数", "desc": "每轮扫描论坛列表的页数，越多越全但越慢", "type": "int"},
    "max_reply_actions":    {"group": "回复参数", "label": "每轮最多回复数", "desc": "每轮回复阶段最多发送的评论数量", "type": "int"},

    # ─── RAG 参数 ────────────────────────────
    "top_k_posts":           {"group": "RAG 参数", "label": "检索相似帖数量", "desc": "生成回复时从数据库检索最相似的 N 个历史帖", "type": "int"},
    "top_k_comments":        {"group": "RAG 参数", "label": "参考高赞评论数量", "desc": "每个相似帖取几条高赞评论作为风格参考", "type": "int"},
    "min_lights":            {"group": "RAG 参数", "label": "参考评论最低赞数", "desc": "只参考赞数 ≥ 此值的评论", "type": "int"},
    "comment_ref_max_chars": {"group": "RAG 参数", "label": "参考评论截断长度", "desc": "注入 Prompt 的参考评论最多取多少字", "type": "int"},
    "post_prompt_max_chars": {"group": "RAG 参数", "label": "正文注入截断长度", "desc": "注入 Prompt 的帖子正文最多取多少字", "type": "int"},
    "current_replies_count": {"group": "RAG 参数", "label": "当前风向评论数", "desc": "抓取帖子时读取最新几条评论作为风向参考", "type": "int"},
    "reply_top_lights":      {"group": "RAG 参数", "label": "读取高赞评论数", "desc": "解析帖子时读取几条高赞评论（lights）", "type": "int"},
    "hybrid_alpha":          {"group": "RAG 参数", "label": "混合检索稠密权重", "desc": "稠密向量得分权重（1-此值=稀疏权重），仅 bge-m3 有效，推荐 0.7", "type": "float"},
    "generate_temperature":  {"group": "RAG 参数", "label": "生成回复温度", "desc": "回复生成的随机性，越高越发散（建议 0.6~0.9）", "type": "float"},
    "max_tokens":            {"group": "RAG 参数", "label": "回复最大 Token 数", "desc": "生成回复的最大长度，过长会被截断", "type": "int"},
    "classify_temperature":  {"group": "RAG 参数", "label": "分类 LLM 温度", "desc": "帖子分类时的 LLM 温度，建议保持低值", "type": "float"},

    # ─── 反思系统 ────────────────────────────
    "memory_positive_threshold":  {"group": "反思系统", "label": "正样本点赞门槛", "desc": "BotComment 赞数 ≥ 此值存为正样本案例", "type": "int"},
    "memory_negative_threshold":  {"group": "反思系统", "label": "负样本点赞上限", "desc": "BotComment 赞数 ≤ 此值存为负样本案例", "type": "int"},
    "reflect_cooldown_hours":     {"group": "反思系统", "label": "案例收集冷静期（小时）", "desc": "发出不足此小时的评论不处理（点赞还不稳定）", "type": "int"},
    "memory_top_k_cases":         {"group": "反思系统", "label": "注入成功案例数", "desc": "生成回复时检索几个最相似的正样本案例注入 Prompt", "type": "int"},
    "memory_top_k_filter":        {"group": "反思系统", "label": "注入冷帖案例数", "desc": "打分阶段注入几个历史冷帖案例标题辅助判断", "type": "int"},
    "memory_case_sim_threshold":  {"group": "反思系统", "label": "案例注入相似度门槛", "desc": "余弦相似度低于此值的案例不注入 Prompt（防止不相关案例乱入），建议 0.7~0.85", "type": "float"},
    "memory_content_max_chars":   {"group": "反思系统", "label": "ReplyContext 正文截断", "desc": "发帖时存储的帖子正文最多保存多少字", "type": "int"},

    # ─── Prompt 模板 ────────────────────────────
    "prompt_generate_system":  {"group": "Prompt 模板", "label": "回复生成：人设 + 风格规则", "desc": "rag_generate 的 system prompt 静态部分（成功案例自动追加在末尾）", "type": "textarea"},
    "prompt_classify_system":  {"group": "Prompt 模板", "label": "帖子打分：评判标准", "desc": "classify_post 的评分准则（历史冷帖案例 + JSON格式要求自动追加）", "type": "textarea"},
    "prompt_cluster_naming":   {"group": "Prompt 模板", "label": "聚类命名：指令", "desc": "给 other 重聚类后的新分类命名（含JSON输出格式，请勿删除）", "type": "textarea"},

    # ─── 防封节奏 ────────────────────────────
    "typing_delay_ms":      {"group": "防封节奏", "label": "打字延迟（毫秒/字）", "desc": "模拟人工打字的每字间隔，越大越安全但越慢", "type": "int"},
    "list_url_base":        {"group": "防封节奏", "label": "列表页基础 URL", "desc": "爬取的论坛列表地址", "type": "str"},
}

# 分组排序
GROUP_ORDER = ["API 设置", "时间控制", "爬虫参数", "分类参数", "回复参数", "RAG 参数", "反思系统", "Prompt 模板", "防封节奏"]


# ══════════════════════════════════════════════
#  路由
# ══════════════════════════════════════════════
@app.route("/")
def index():
    # 读取当前覆盖值
    overrides = {}
    try:
        with open(OVERRIDES_FILE, encoding="utf-8") as f:
            overrides = json.load(f)
    except Exception:
        pass

    # 按分组整理参数
    groups: dict[str, list] = {g: [] for g in GROUP_ORDER}
    groups["其他"] = []
    for key, val in CONFIG.items():
        meta = PARAM_META.get(key)
        if meta is None:
            continue
        group = meta.get("group", "其他")
        if group not in groups:
            groups[group] = []
        current = overrides.get(key, val)
        groups[group].append({
            "key":     key,
            "label":   meta["label"],
            "desc":    meta["desc"],
            "type":    meta["type"],
            "default": val,
            "value":   current,
        })

    # 生成标签页 HTML
    tabs_nav, tabs_content = "", ""
    for i, (group, params) in enumerate(groups.items()):
        if not params:
            continue
        tab_id  = f"tab-{i}"
        active  = "active" if i == 0 else ""
        show    = "show active" if i == 0 else ""
        tabs_nav += f'<button class="nav-link {active}" data-bs-toggle="tab" data-bs-target="#{tab_id}">{group}</button>\n'

        rows = ""
        for p in params:
            val = p["value"]
            key = p["key"]
            if p["type"] == "bool":
                checked = "checked" if val else ""
                input_html = f'<input type="checkbox" class="form-check-input param-input" id="{key}" name="{key}" {checked} data-type="bool">'
                rows += f'''
                <tr>
                  <td class="fw-semibold text-nowrap">{p["label"]}</td>
                  <td><div class="form-check">{input_html}</div></td>
                  <td class="text-muted small">{p["desc"]}</td>
                  <td class="text-muted small font-monospace">默认: {p["default"]}</td>
                </tr>'''
            elif p["type"] == "password":
                rows += f'''
                <tr>
                  <td class="fw-semibold text-nowrap">{p["label"]}</td>
                  <td><input type="password" class="form-control form-control-sm param-input" id="{key}" name="{key}" value="{val}" data-type="str"></td>
                  <td class="text-muted small">{p["desc"]}</td>
                  <td class="text-muted small font-monospace">默认: {'*' * 8}</td>
                </tr>'''
            elif p["type"] == "textarea":
                escaped = _html.escape(str(val))
                rows += f'''
                <tr>
                  <td class="fw-semibold text-nowrap align-top pt-2">{p["label"]}</td>
                  <td colspan="3">
                    <textarea class="form-control form-control-sm param-input font-monospace" id="{key}" name="{key}" rows="7" data-type="str" style="white-space:pre;font-size:0.8rem">{escaped}</textarea>
                    <div class="text-muted small mt-1">{p["desc"]}</div>
                  </td>
                </tr>'''
            else:
                rows += f'''
                <tr>
                  <td class="fw-semibold text-nowrap">{p["label"]}</td>
                  <td><input type="text" class="form-control form-control-sm param-input" id="{key}" name="{key}" value="{val}" data-type="{p["type"]}"></td>
                  <td class="text-muted small">{p["desc"]}</td>
                  <td class="text-muted small font-monospace">默认: {p["default"]}</td>
                </tr>'''

        tabs_content += f'''
        <div class="tab-pane fade {show}" id="{tab_id}">
          <div class="table-responsive mt-3">
            <table class="table table-hover align-middle">
              <thead class="table-light">
                <tr><th style="width:22%">参数</th><th style="width:20%">当前值</th><th>说明</th><th style="width:18%">默认值</th></tr>
              </thead>
              <tbody>{rows}</tbody>
            </table>
          </div>
        </div>'''

    # ── 记事本标签页 ──────────────────────────────────
    notes_content = ""
    try:
        with open("data/notes.md", encoding="utf-8") as _nf:
            notes_content = _nf.read()
    except FileNotFoundError:
        pass
    notes_escaped = _html.escape(notes_content)
    notes_tab_id  = "tab-notes"
    tabs_nav     += f'<button class="nav-link" data-bs-toggle="tab" data-bs-target="#{notes_tab_id}">📝 记事本</button>\n'
    tabs_content += f'''
    <div class="tab-pane fade" id="{notes_tab_id}">
      <div class="mt-3">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <span class="text-muted small">自由记录想做的功能、当前不满意的地方等，保存到 data/notes.md</span>
          <button class="btn btn-success btn-sm" onclick="saveNotes()">💾 保存记事本</button>
        </div>
        <textarea id="notes-area" class="form-control font-monospace" rows="30"
          style="font-size:0.88rem; white-space:pre; resize:vertical">{notes_escaped}</textarea>
      </div>
    </div>'''

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>虎扑 Bot 参数配置</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
  <style>
    body {{ background: #f8f9fa; }}
    .navbar {{ background: #1a1a2e !important; }}
    .nav-pills .nav-link.active {{ background-color: #0d6efd; }}
    .toast-container {{ position: fixed; top: 1rem; right: 1rem; z-index: 9999; }}
    td {{ vertical-align: middle !important; }}
    .param-input {{ min-width: 120px; }}
    .modified {{ background-color: #fff3cd !important; }}
  </style>
</head>
<body>
<nav class="navbar navbar-dark mb-4">
  <div class="container-fluid">
    <span class="navbar-brand fw-bold">🐯 虎扑 Bot 参数配置</span>
    <div class="d-flex gap-2">
      <button class="btn btn-outline-warning btn-sm" onclick="resetAll()">重置为默认</button>
      <button class="btn btn-success" onclick="saveAll()">💾 保存</button>
    </div>
  </div>
</nav>

<div class="container-fluid px-4">
  <div class="toast-container">
    <div id="toast" class="toast align-items-center text-bg-success border-0" role="alert">
      <div class="d-flex">
        <div class="toast-body" id="toast-msg">已保存</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
      </div>
    </div>
  </div>

  <ul class="nav nav-pills flex-wrap gap-1 mb-2">
    {tabs_nav}
  </ul>

  <div class="tab-content">
    {tabs_content}
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
function collectValues() {{
  const result = {{}};
  document.querySelectorAll('.param-input').forEach(el => {{
    const t = el.dataset.type;
    if (t === 'bool') result[el.name] = el.checked;
    else if (t === 'int') result[el.name] = parseInt(el.value) || 0;
    else if (t === 'float') result[el.name] = parseFloat(el.value) || 0;
    else result[el.name] = el.value;
  }});
  return result;
}}

function saveAll() {{
  fetch('/save', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(collectValues())
  }})
  .then(r => r.json())
  .then(d => showToast(d.ok ? '✅ 已保存' : '❌ 保存失败: ' + d.error, d.ok ? 'success' : 'danger'));
}}

function saveNotes() {{
  fetch('/notes_save', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{content: document.getElementById('notes-area').value}})
  }})
  .then(r => r.json())
  .then(d => showToast(d.ok ? '✅ 记事本已保存' : '❌ 保存失败: ' + d.error, d.ok ? 'success' : 'danger'));
}}

function resetAll() {{
  if (!confirm('确认重置所有参数为默认值？')) return;
  fetch('/reset', {{method: 'POST'}})
  .then(r => r.json())
  .then(d => {{ if(d.ok) location.reload(); }});
}}

function showToast(msg, type) {{
  const t = document.getElementById('toast');
  t.className = `toast align-items-center text-bg-${{type}} border-0`;
  document.getElementById('toast-msg').textContent = msg;
  new bootstrap.Toast(t, {{delay: 2500}}).show();
}}

// 高亮已修改项
document.querySelectorAll('.param-input').forEach(el => {{
  const orig = el.type === 'checkbox' ? el.checked : el.value;
  el.addEventListener('change', () => {{
    const cur = el.type === 'checkbox' ? el.checked : el.value;
    el.closest('tr').classList.toggle('modified', cur != orig);
  }});
}});
</script>
</body>
</html>"""
    return html


@app.route("/save", methods=["POST"])
def save():
    try:
        data = request.get_json()
        with open(OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/reset", methods=["POST"])
def reset():
    try:
        if os.path.exists(OVERRIDES_FILE):
            os.remove(OVERRIDES_FILE)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/notes_save", methods=["POST"])
def notes_save():
    try:
        content = request.get_json().get("content", "")
        with open("data/notes.md", "w", encoding="utf-8") as f:
            f.write(content)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ══════════════════════════════════════════════
#  启动
# ══════════════════════════════════════════════
def open_edge(port: int):
    time.sleep(1.2)
    url = f"http://localhost:{port}"
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for path in edge_paths:
        if os.path.exists(path):
            subprocess.Popen([path, url])
            return
    # 备用：系统默认浏览器
    webbrowser.open(url)


if __name__ == "__main__":
    PORT = 5174
    print(f"🌐 配置编辑器启动中，请稍候...")
    print(f"   地址: http://localhost:{PORT}")
    print(f"   修改后点击「保存」写入 data/config_overrides.json")
    print(f"   关闭终端窗口即可停止服务\n")
    threading.Thread(target=open_edge, args=(PORT,), daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, debug=False)
