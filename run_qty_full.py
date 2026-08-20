# -*- coding: utf-8 -*-
"""全量数量异常检测：清洗→特征→白名单→Q1/Q3规则+KDE打分→全量打标签（异常+正常一起输出）"""
import pandas as pd, numpy as np, os, time, warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_process import data_process, feature_engineer
from rule_engine import RuleEngine, register_all_rules
from detectors.qty_detector import qty_detector
from whitelist import load_whitelist, apply_whitelist

DATA_FILE = os.path.join(BASE_DIR, '内采订单商品信息明细数据（20260101-0622）.xlsx')
OUT_DIR = os.path.join(BASE_DIR, '类别专项检测')
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 50)
print("全量数量异常检测")
print("=" * 50)

print("[1] 读取数据...")
df = pd.read_excel(DATA_FILE)
print(f"    原始 {len(df)} 行")

print("[2] 数据清洗...")
df = data_process(df)
print("[3] 特征工程...")
df = feature_engineer(df)

print("[4] 白名单...")
wl, wl_set = load_whitelist()
df = apply_whitelist(df, wl, wl_set)
print(f"    白名单实体: {len(wl_set)} 个")

print("[5] 数量检测器（Q1/Q3规则 + per-SKU KDE）...")
engine = RuleEngine()
register_all_rules(engine)
df = qty_detector(df, engine)

# 白名单行强制无异常
wl_mask = df['is_whitelisted'] == 1
df.loc[wl_mask, 'qty_score'] = 0.0
df.loc[wl_mask, 'qty_rule_reason'] = ''

# 打标签（与 fusion 判定一致：数量维度得分 >= 90 分即异常，白名单除外）
df['qty_label'] = np.where(df['qty_score'] >= BINARY_THRESHOLD, '异常', '无异常')

ts = int(time.time())
out_cols = [
    COL_ORDER_ID, COL_ORD_ITEM_ID, COL_SUBMIT_TIME,
    COL_SMTR_NAME, COL_DEPT, COL_SUP_NAME, COL_SKU_NAME,
    COL_FOUR_CAT, COL_MATERIAL_TYPE, COL_CONSIGNER, COL_PROJ_NAME,
    COL_TAX_PRICE, COL_PUR_QTY, COL_SUB_TTL,
    'qty_score', 'qty_rule_reason', 'qty_label', 'whitelist_reason',
]
out_cols = [c for c in out_cols if c in df.columns]

out_path = os.path.join(OUT_DIR, f'全量数量异常检测结果_{ts}.xlsx')
df[out_cols].to_excel(out_path, index=False)

total = len(df)
anom = (df['qty_label'] == '异常').sum()
normal = (df['qty_label'] == '无异常').sum()
print(f"\n[6] 完成")
print(f"    总行数: {total} | 异常: {anom} ({anom/total*100:.2f}%) | 无异常: {normal} ({normal/total*100:.2f}%)")
if 'qty_rule_reason' in df.columns:
    print(f"    命中来源分布:")
    reasons = df.loc[df['qty_label'] == '异常', 'qty_rule_reason']
    for r, c in reasons.value_counts().items():
        print(f"      {r}: {c}")
print(f"    结果: {out_path}")
