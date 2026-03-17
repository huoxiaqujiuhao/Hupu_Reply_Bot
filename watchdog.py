"""
watchdog.py — 启动 main.py 并全程监控
- 每 5 分钟检查一次回帖速率
- 30 分钟内回帖超过阈值 → 杀进程
- main.py 正常结束或被杀死后，自动打印诊断摘要
"""
import subprocess
import sys
import os
import re
import time
from datetime import datetime, timedelta

os.makedirs("data/logs", exist_ok=True)
WATCHDOG_LOG = "data/logs/watchdog.log"
HUPU_LOG     = "data/logs/hupu.log"

MAX_REPLIES_30MIN = 12   # 30 分钟内超过这个数 → 杀进程
CHECK_INTERVAL    = 300  # 每 5 分钟检查一次（秒）


def wlog(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [WATCHDOG] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_replies_in_window(minutes: int = 30) -> int:
    """统计最近 N 分钟日志里出现"发送成功"的次数。"""
    try:
        cutoff = datetime.now() - timedelta(minutes=minutes)
        count  = 0
        with open(HUPU_LOG, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.match(r'(\d{2}):(\d{2}):(\d{2})', line)
                if m and "发送成功" in line:
                    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    t = datetime.now().replace(hour=h, minute=mi, second=s, microsecond=0)
                    # 跨午夜修正
                    if t > datetime.now():
                        t -= timedelta(days=1)
                    if t >= cutoff:
                        count += 1
        return count
    except Exception:
        return 0


def tail_log(lines: int = 60) -> str:
    """读取日志最后 N 行。"""
    try:
        with open(HUPU_LOG, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except Exception:
        return "(日志读取失败)"


def summarize(reason: str):
    wlog("=" * 50)
    wlog(f"终止原因：{reason}")
    wlog("最近 60 行日志：")
    for line in tail_log(60).splitlines():
        wlog("  " + line)
    wlog("=" * 50)


def main():
    wlog("Watchdog 启动，拉起 main.py ...")
    proc = subprocess.Popen([sys.executable, "main.py"])
    wlog(f"main.py PID = {proc.pid}")


    while proc.poll() is None:
        time.sleep(CHECK_INTERVAL)

        if proc.poll() is not None:
            break

        n = count_replies_in_window(30)
        wlog(f"过去 30 分钟回帖数：{n}（阈值 {MAX_REPLIES_30MIN}）")

        if n > MAX_REPLIES_30MIN:
            wlog(f"⚠️  回帖过快！{n} 条 / 30min，正在杀进程...")
            proc.terminate()
            time.sleep(3)
            if proc.poll() is None:
                proc.kill()
            summarize(f"回帖速率超限（{n} 条/30min > {MAX_REPLIES_30MIN}）")
            sys.exit(1)

    code = proc.wait()
    if code == 0:
        wlog("main.py 正常结束（exit 0）")
    else:
        wlog(f"main.py 异常退出（exit {code}）")
        summarize(f"main.py 非零退出码 {code}")


if __name__ == "__main__":
    main()
