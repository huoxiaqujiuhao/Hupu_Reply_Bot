"""
basketball_analyze.py — 分析篮球区语料，统计高频球星 / 关键词

运行方式：python basketball_analyze.py
需要先运行 basketball_scraper.py 采集足够数据。
"""
import sqlite3
import re
from collections import Counter
from config import CONFIG

# 候选球星名（宽泛，覆盖全名/简称/英文名/绰号）
CANDIDATE_NAMES = [
    "詹姆斯", "勒布朗", "LBJ",
    "库里", "斯蒂芬",
    "科比", "布莱恩特",
    "杜兰特", "KD",
    "字母哥", "安特托昆博",
    "欧文", "凯里",
    "哈登", "胡子",
    "威少", "威斯布鲁克",
    "保罗", "CP3",
    "浓眉", "戴维斯",
    "约基奇",
    "字母",
    "东契奇", "卢卡",
    "塔图姆",
    "布克",
    "米切尔",
    "福克斯",
    "文班亚马",
    "艾弗森", "AI",
    "乔丹", "MJ",
    "奥尼尔", "鲨鱼",
    "加内特",
    "麦迪",
    "诺维茨基",
    "韦德",
    "波什",
    "巴特勒",
    "厄文",
]

def analyze():
    conn = sqlite3.connect(CONFIG["db_name"])
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM Posts WHERE section='basketball'")
    total = cur.fetchone()[0]
    print(f"\n篮球区帖子总数：{total}\n")

    if total == 0:
        print("数据库里还没有篮球区帖子，请先运行 basketball_scraper.py")
        conn.close()
        return

    # 按回复数统计
    cur.execute("""
        SELECT total_replies FROM Posts WHERE section='basketball'
        ORDER BY total_replies DESC
    """)
    replies = [r[0] for r in cur.fetchall()]
    print(f"回复数分布：")
    print(f"  最高：{replies[0]}  最低：{replies[-1]}  中位：{replies[len(replies)//2]}")
    print(f"  200+：{sum(1 for r in replies if r >= 200)} 条")
    print(f"  500+：{sum(1 for r in replies if r >= 500)} 条")
    print(f"  1000+：{sum(1 for r in replies if r >= 1000)} 条\n")

    # 球星名频率统计
    cur.execute("SELECT title FROM Posts WHERE section='basketball'")
    titles = [r[0] for r in cur.fetchall()]
    all_titles = " ".join(titles)

    print("球星名出现频率（按标题统计）：")
    print(f"  {'名字':<12} {'出现次数':>8}  {'占比':>6}")
    print("  " + "-" * 32)

    counts = []
    for name in CANDIDATE_NAMES:
        n = len(re.findall(re.escape(name), all_titles))
        if n > 0:
            counts.append((name, n))

    counts.sort(key=lambda x: -x[1])
    for name, n in counts:
        pct = n / len(titles) * 100
        print(f"  {name:<12} {n:>8}  {pct:>5.1f}%")

    # 找未在候选名单里的高频词（可能有漏网球星）
    print("\n\n高频双字词/三字词（可能有遗漏球星）：")
    words = re.findall(r'[\u4e00-\u9fa5]{2,4}', all_titles)
    # 过滤掉已在候选名单里的
    known = set(CANDIDATE_NAMES)
    counter = Counter(w for w in words if w not in known and len(w) >= 2)
    print("  （排除已知球星名后）")
    for word, cnt in counter.most_common(30):
        if cnt >= 3:
            print(f"  {word}  {cnt}")

    conn.close()


if __name__ == "__main__":
    analyze()
