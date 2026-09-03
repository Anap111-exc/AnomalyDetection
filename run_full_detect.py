# -*- coding: utf-8 -*-
"""
全品类完整异常检测（快速模式）
- 在全部数据上一次性运行完整检测流程（规则 + 价格/数量 KDE + 拆单 + 聚量倒V）
- 快速模式默认跳过 Prophet 时序与图算法提权（可按需开启：USE_PROPHET=1 python run_full_detect.py）
- 输出：
  1) 全品类-全量标注.xlsx   ：全部订单 + 异常标注（含所有异常类型）
  2) 全品类-汇总.xlsx       ：按四级品类的异常统计
用法：
  USE_PROPHET=0 python run_full_detect.py          # 快速（推荐全品类）
  USE_PROPHET=1 python run_full_detect.py          # 含 Prophet（SKU 多时很慢）
"""
import pandas as pd, numpy as np, io, sys, os, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from config import *
from data_process import data_process, feature_engineer
from rule_engine import RuleEngine, register_all_rules
from detectors import price_detector, qty_detector, split_detector, concen_detector
from fusion import compute_final_score
from whitelist import load_whitelist, apply_whitelist
import glob

def locate_data_file():
    if INPUT_FILE:
        p = os.path.join(DATA_DIR, INPUT_FILE)
        if os.path.exists(p):
            return p
        raise FileNotFoundError(f"指定的数据文件不存在: {p}")
    for ext in ("*.csv", "*.xlsx"):
        files = sorted(glob.glob(os.path.join(DATA_DIR, ext)))
        files = [f for f in files if "内采异常数据分析" not in os.path.basename(f)]
        if files:
            return files[0]
    raise FileNotFoundError(f"未在 {DATA_DIR} 找到数据文件")

# 读数据时按类型压缩内存（字符串字段显式 str/category；数值列不强制类型，读取后容错转换，
# 避免个别文件列含脏文本时 dtype 强转直接崩溃）
FULL_DTYPE = {
    'order_id': 'str', 'order_no': 'str', 'ord_item_id': 'str',
    'sku_code': 'category', 'sku_name': 'category',
    'one_cat_name': 'category', 'two_cat_name': 'category',
    'three_cat_name': 'category', 'four_cat_name': 'category',
    'sup_short_name': 'category', 'company_name': 'category', 'store_name': 'category',
    'material_type_name': 'category', 'rec_address': 'category',
    'smtr_name': 'category', 'thd_dept_short_name': 'category',
    'sec_dept_short_name': 'category', 'consigner_name': 'category', 'proj_name': 'category',
    'ord_type_name': 'str', 'ord_status_name': 'str',
    'sec_dept_type': 'str', 'material_type': 'str', 'store_pzn_level': 'str',
    'is_povt_alevt': 'str', 'measure_unit_name': 'str', 'apply_people_name': 'str',
}
# 数值列（读取后统一转数值，脏值转 NaN 由清洗阶段处理）
NUMERIC_COLS = ['pur_qty', 'origin_tax_price', 'tax_price', 'sub_ttl']

print(f"读取数据: {locate_data_file()}", flush=True)
raw_path = locate_data_file()
df_all = pd.read_csv(raw_path, dtype=FULL_DTYPE) if raw_path.endswith(".csv") else pd.read_excel(raw_path)
for c in NUMERIC_COLS:
    if c in df_all.columns:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
df_all = data_process(df_all)
df_all = feature_engineer(df_all)
print(f"全量清洗后: {len(df_all)} 行, {df_all[COL_FOUR_CAT].nunique()} 个四级品类", flush=True)
print(f"内存占用: {df_all.memory_usage(deep=True).sum()/1024**2:.0f} MB", flush=True)

wl, wl_set = load_whitelist()
df_all = apply_whitelist(df_all, wl, wl_set)
engine = RuleEngine(); register_all_rules(engine)
df_all = engine.execute(df_all)
df_all = price_detector(df_all, engine)
df_all = qty_detector(df_all, engine)
df_all = split_detector(df_all, engine)
df_all = concen_detector(df_all, engine)
print("(快速模式跳过图算法提权)", flush=True)

df_all = compute_final_score(df_all)

os.makedirs(RESULT_ROOT, exist_ok=True)
out_full = os.path.join(RESULT_ROOT, "全品类-全量标注.xlsx")
cols = [c for c in ["order_id", "order_no", "ord_item_id", "submit_time",
        "smtr_name", "thd_dept_short_name", "sec_dept_short_name", "sup_short_name",
        "consigner_name", "rec_address", "sku_name", "proj_name", "four_cat_name",
        "tax_price", "pur_qty", "sub_ttl",
        "anomaly_type", "risk_level",
        "price_type", "qty_type", "split_type", "concen_type"] if c in df_all.columns]
df_all[cols].to_excel(out_full, index=False)
print(f"输出: {out_full} | 总{len(df_all)}行 异常{(df_all['risk_level']=='异常').sum()}行", flush=True)

summ = df_all[df_all["risk_level"] == "异常"].groupby(COL_FOUR_CAT)["anomaly_type"] \
    .agg(lambda s: s.value_counts().to_dict()).reset_index()
summ.columns = ["品类", "异常构成"]
summ.insert(1, "异常行数", df_all[df_all["risk_level"] == "异常"].groupby(COL_FOUR_CAT).size().values)
summ = summ.sort_values("异常行数", ascending=False)
out_summ = os.path.join(RESULT_ROOT, "全品类-汇总.xlsx")
summ.to_excel(out_summ, index=False)
print(f"输出: {out_summ} | 共 {len(summ)} 个品类有异常", flush=True)
print("全部完成!")