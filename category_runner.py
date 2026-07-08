# -*- coding: utf-8 -*-
"""专类检测：茶叶+打印机耗材全量检测 + 全量拆单检测"""
import pandas as pd, numpy as np, os, time, warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_process import data_process, feature_engineer
from rule_engine import RuleEngine, register_all_rules
from detectors import (
    price_detector, qty_detector, split_detector,
    concen_detector, invis_detector,
)
from graph.build_graph import build_dim_graph
from graph.graph_algo import run_louvain, run_pagerank
from graph.boost import compute_boost, map_boost_to_orders
from fusion import compute_final_score
from whitelist import load_whitelist, apply_whitelist
from audit import generate_audit_report

# 目标类目关键词
TEA_KEYWORDS = ['茶']
PRINTER_KEYWORDS = ['打印', '复印', '耗材']

# 结果输出目录
OUT_DIR = os.path.join(BASE_DIR, '类别专项检测')
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Part 1: 全量拆单检测 ======
print("=" * 50)
print("Part 1: 全量拆单检测")
print("=" * 50)

df_all = pd.read_excel(INPUT_FILE)
df_all = data_process(df_all)
df_all = feature_engineer(df_all)

engine = RuleEngine()
register_all_rules(engine)

df_split = df_all.copy()
if 'split_score' not in df_split.columns:
    df_split['split_score'] = 0.0
df_split = engine.execute(df_split, dim='split')

split_hit = df_split[df_split['split_score'] > 0]
print(f"\n全量拆单命中: {len(split_hit)} 行")
cols = ['order_id','submit_time','smtr_name','sup_short_name','sku_name','tax_price','pur_qty','sub_ttl',
        'split_score','split_type','consigner_name','rec_address','proj_name']
cols = [c for c in cols if c in split_hit.columns]
split_hit[cols].to_excel(os.path.join(OUT_DIR, '全量拆单检测.xlsx'), index=False)
print(f"全量拆单结果: {os.path.join(OUT_DIR, '全量拆单检测.xlsx')}")

# ====== Part 2: 茶叶+打印机耗材全量检测 ======
print("\n" + "=" * 50)
print("Part 2: 茶叶+打印机耗材全量检测")
print("=" * 50)

# 筛选目标类目
mask_tea = df_all[COL_FOUR_CAT].str.contains('|'.join(TEA_KEYWORDS), na=False)
mask_printer = df_all[COL_FOUR_CAT].str.contains('|'.join(PRINTER_KEYWORDS), na=False)
df_cat = df_all[mask_tea | mask_printer].copy()
print(f"茶叶: {mask_tea.sum()} 行 | 打印机耗材: {mask_printer.sum()} 行 | 合计: {len(df_cat)} 行")

# 白名单
whitelist, wl_set = load_whitelist()
df_cat = apply_whitelist(df_cat, whitelist, wl_set)

# 5个检测器
df_cat = engine.execute(df_cat)  # all rules
df_cat = price_detector(df_cat, engine)
df_cat = qty_detector(df_cat, engine)
df_cat = split_detector(df_cat, engine)
df_cat = concen_detector(df_cat, engine)
df_cat = invis_detector(df_cat, engine)

# 图算法
graph_dims = ['price', 'qty', 'concen', 'invis']
for dim in graph_dims:
    try:
        G = build_dim_graph(df_cat, dim, whitelist=wl_set)
        if G.number_of_nodes() == 0:
            continue
        partition, communities = run_louvain(G)
        pr = run_pagerank(G)
        node_scores = {}
        score_col = f'{dim}_score'
        if score_col in df_cat.columns:
            for _, row in df_cat.iterrows():
                for entity in [str(row.get(COL_SMTR_NAME, '')), str(row.get(COL_SUP_NAME, '')), str(row.get(COL_SKU_NAME, ''))]:
                    if entity:
                        node_scores[entity] = max(node_scores.get(entity, 0), row[score_col])
        node_boost = compute_boost(partition, pr, communities, node_scores, whitelist=wl_set)
        df_cat = map_boost_to_orders(df_cat, node_boost, dim)
    except Exception as e:
        print(f"  [{dim}] 图失败: {e}")
        df_cat[f'graph_{dim}_boost'] = 0

# 决策
df_cat = compute_final_score(df_cat)

# 输出
ts = int(time.time())
anomaly_cols = [COL_ORDER_ID, COL_ORD_ITEM_ID, COL_SUBMIT_TIME,
                COL_SMTR_NAME, COL_DEPT, COL_SUP_NAME, COL_CONSIGNER, COL_REC_ADDRESS,
                COL_SKU_NAME, COL_PROJ_NAME, COL_FOUR_CAT,
                COL_TAX_PRICE, COL_PUR_QTY, COL_SUB_TTL,
                'price_score', 'qty_score', 'split_score', 'concen_score', 'invis_score',
                'anomaly_type', 'anomaly_detail', 'risk_level']
anomaly_cols += [f'graph_{dim}_boost' for dim in graph_dims]
anomaly_cols = [c for c in anomaly_cols if c in df_cat.columns]
out_path = os.path.join(OUT_DIR, f'茶叶_打印机耗材_全量检测_{ts}.xlsx')
df_cat[anomaly_cols].to_excel(out_path, index=False)

print(f"\n=== 检测结果 ===")
total_rows = len(df_cat)
for lvl in ['异常', '无异常']:
    cnt = (df_cat['risk_level'] == lvl).sum()
    print(f"  {lvl}: {cnt} ({cnt/total_rows*100:.1f}%)")
print(f"  异常类型分布:")
for t, c in df_cat['anomaly_type'].value_counts().head(8).items():
    print(f"    {t}: {c}")
print(f"\n茶叶+打印机耗材结果: {out_path}")

# 审计Top-50
audit_df = df_cat[df_cat['risk_level'] == '异常']
if len(audit_df) > 0:
    boosted = [f'boosted_{d}' for d in ['price','qty','split','concen','invis'] if f'boosted_{d}' in df_cat.columns]
    if boosted:
        df_cat['_sort'] = df_cat[boosted].max(axis=1)
        audit_top = df_cat.nlargest(50, '_sort')
    else:
        audit_top = audit_df.head(50)
    audit_top[anomaly_cols].to_excel(os.path.join(OUT_DIR, f'audit_top50_{ts}.xlsx'), index=False)

print(f"\n全部完成！结果在: {OUT_DIR}")
