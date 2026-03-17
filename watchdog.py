"""
watchdog.py -- 启动 main.py 并全程监控
- 每 5 分钟检查一次回帖速率
- 30 分钟内回帖超过阈值 -> 杀进程
- main.py 正常结束或被杀死后，自动打印诊断摘要
"""
import subprocess
import sys
import io
import os
import re
import time
from datetime import datetime, timedelta

# 修复 Windows GBK 终端乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.makedirs("data/logs", exist_ok=True)
WATCHDOG_LOG = "data/logs/watchdog.log"
HUPU_LOG     = "data/logs/hupu.log"

MAX_REPLIES_30MIN = 12   # 30 分钟内超过这个数 -> 杀进程
CHECK_INTERVAL    = 300  # 每 5 分钟检查一次（秒）

# watchdog 自身启动时间，用于只统计本次运行以来的回帖
_START_TIME = datetime.now()


def wlog(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} [WATCHDOG] {msg}"
    print(line, flush=True)
    with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_replies_since_start() -> int:
    """统计 watchdog 启动以来日志里出现'发送成功'的次数（只统计本次运行后的条目）。"""
    try:
        count = 0
        # 只读日志最后 5000 行，避免扫全文
        with open(HUPU_LOG, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for line in lines[-5000:]:
            if "发送成功" not in line:
                continue
            m = re.match(r'(\d{2}):(\d{2}):(\d{2})', line)
            if not m:
                continue
            h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            t = datetime.now().replace(hour=h, minute=mi, second=s, microsecond=0)
            if t > datetime.now():          # 跨午夜修正
                t -= timedelta(days=1)
            if t >= _START_TIME:            # 只统计本次 watchdog 启动后的回帖
                count += 1
        return count
    except Exception:
        return 0


def tail_log(lines: int = 60) -> str:
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


def kill_proc(proc):
    proc.terminate()
    time.sleep(3)
    if proc.poll() is None:
        proc.kill()


def main():
    wlog("Watchdog 启动，拉起 main.py ...")
    proc = subprocess.Popen([sys.executable, "main.py"])
    wlog(f"main.py PID = {proc.pid}")

    while proc.poll() is None:
        time.sleep(CHECK_INTERVAL)

        if proc.poll() is not None:
            break

        n = count_replies_since_start()
        elapsed = (datetime.now() - _START_TIME).total_seconds() / 60
        wlog(f"本次运行已回帖 {n} 条（运行 {elapsed:.0f} 分钟，阈值 {MAX_REPLIES_30MIN} 条/30min）")

        # 按速率判断：elapsed 分钟内发了 n 条，换算到 30 分钟的预期
        if elapsed > 5 and n / elapsed * 30 > MAX_REPLIES_30MIN:
            rate = n / elapsed * 30
            wlog(f"[ALERT] 回帖速率过高！折算 30min 约 {rate:.1f} 条，正在杀进程...")
            kill_proc(proc)
            summarize(f"回帖速率超限（折算 {rate:.1f} 条/30min > {MAX_REPLIES_30MIN}）")
            sys.exit(1)

    code = proc.wait()
    if code == 0:
        wlog("main.py 正常结束（exit 0）")
    else:
        wlog(f"main.py 异常退出（exit {code}）")
        summarize(f"main.py 非零退出码 {code}")


if __name__ == "__main__":
    main()
