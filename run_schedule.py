# -*- coding: utf-8 -*-
"""
定时检测脚本（配合 cron 使用）
用法：
  python run_schedule.py "茶叶,牛奶（不含鲜奶）"      # 指定品类
  python run_schedule.py                             # 用默认品类列表
输出：/home/tione/notebook/dataresult/品类专项/<品类名>-内采异常数据分析.xlsx
"""
import sys, os, io, time, logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from config import *
import detect_categories as dc

DEFAULT_CATS = ["茶叶", "牛奶（不含鲜奶）", "标签机", "排插插排", "显示器支架", "打印机及复印机配件"]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main():
    if len(sys.argv) >= 2:
        cats = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    else:
        cats = DEFAULT_CATS
    log(f"定时任务启动，待检测品类: {cats}")
    log(f"数据文件: {os.path.join(DATA_DIR, INPUT_FILE) if INPUT_FILE else '自动定位'}")
    t0 = time.time()
    results = dc.detect(",".join(cats))
    for cat, path in results.items():
        log(f"完成: {cat} -> {path}")
    log(f"全部完成，总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()