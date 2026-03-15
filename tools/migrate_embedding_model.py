"""
migrate_embedding_model.py — 切换 Embedding 模型后的一键迁移

使用方法：
  1. 先在 config.py 或图形界面里改好 embedding_model
  2. python tools/migrate_embedding_model.py

做了什么：
  - 删除 data/centroids.json（下次 main.py 启动时自动从已打标 Posts 重建）
  - 重置所有 BotComments.reflected=0（让反思系统用新模型重新向量化案例）
  - 清空 Cases 表的所有记录（旧 embedding 维度不兼容，需重算）
  - 删除旧的 Memories 表（已废弃的规则系统残留）
"""
import os
import sqlite3
from config import CONFIG

DB   = CONFIG["db_name"]
CENT = CONFIG["centroids_file"]

# 1. 删质心文件
if os.path.exists(CENT):
    os.remove(CENT)
    print(f"✅ 已删除 {CENT}")
else:
    print(f"ℹ️  {CENT} 不存在，跳过")

# 2. 数据库操作
conn = sqlite3.connect(DB)
cur  = conn.cursor()

# 重置 reflected（如果列存在）
try:
    cur.execute("UPDATE BotComments SET reflected=0")
    n = cur.rowcount
    print(f"✅ 重置 BotComments.reflected=0，共 {n} 条")
except Exception as e:
    print(f"ℹ️  BotComments.reflected 列尚不存在（会在启动时自动添加），跳过：{e}")

# 清空 Cases（如果存在）
try:
    cur.execute("DELETE FROM Cases")
    n = cur.rowcount
    print(f"✅ 清空 Cases 表，共删除 {n} 条")
except Exception:
    print("ℹ️  Cases 表不存在，跳过")

# 删掉旧 Memories 表
try:
    cur.execute("DROP TABLE IF EXISTS Memories")
    print("✅ 已删除旧 Memories 表")
except Exception as e:
    print(f"⚠️  删除 Memories 表失败：{e}")

conn.commit()
conn.close()

print("\n🎉 迁移完成。请确认 config.py 里的 embedding_model 已改好，然后正常启动 main.py 即可。")
print("   centroids 会在启动时从已打标帖子自动重建，Cases 会在反思阶段重新填充。")
