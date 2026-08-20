# -*- coding: utf-8 -*-
"""柜式空调/挂式空调 → four_cat_name=='家用空调' 全检测"""
import sys, os, glob, warnings, io, time
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
BASE = r"C:\Users\Administrator\Desktop\python代码\虚拟环境1-测试\代码和结果6"
sys.path.insert(0, BASE); os.chdir(BASE)
import logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

import pandas as pd, numpy as np
from config import *
from data_process import data_process, feature_engineer
from rule_engine import RuleEngine, register_all_rules
from detectors import price_detector, qty_detector, split_detector, concen_detector, invis_detector
from graph.build_graph import build_dim_graph
from graph.graph_algo import run_louvain, run_pagerank
from graph.boost import compute_boost, map_boost_to_orders
from fusion import compute_final_score
from whitelist import load_whitelist, apply_whitelist

CAT_NAME = '油色谱在线监测装置维护'
KW = '油色谱在线监测装置维护'
MODE = 'exact'

raw = glob.glob(os.path.join(BASE, "内采订单商品信息明细数据*.xlsx"))[0]
df_all = data_process(pd.read_excel(raw)); df_all = feature_engineer(df_all)
wl, wl_set = load_whitelist()

t0 = time.time()
if MODE == 'exact':
    cat = df_all[df_all[COL_FOUR_CAT] == KW].copy()
else:
    cat = df_all[df_all[COL_FOUR_CAT].astype(str).str.contains(KW, na=False)].copy()
print(f"品类: {CAT_NAME} (口径: {MODE} '{KW}')  {len(cat)} 行, {cat['sku_name'].nunique()} SKU", flush=True)

cat = apply_whitelist(cat, wl, wl_set)
engine = RuleEngine(); register_all_rules(engine)
cat = engine.execute(cat)
cat = price_detector(cat, engine)
cat = qty_detector(cat, engine)
cat = split_detector(cat, engine)
cat = concen_detector(cat, engine)

for dim in ['price', 'qty', 'concen']:
    try:
        G = build_dim_graph(cat, dim, whitelist=wl_set)
        if G.number_of_nodes() == 0:
            continue
        partition, communities = run_louvain(G)
        pr = run_pagerank(G)
        node_scores = {}
        score_col = f'{dim}_score'
        if score_col in cat.columns:
            for _, row in cat.iterrows():
                for entity in [str(row.get(COL_SMTR_NAME, '')), str(row.get(COL_SUP_NAME, '')), str(row.get(COL_SKU_NAME, ''))]:
                    if entity:
                        node_scores[entity] = max(node_scores.get(entity, 0), row[score_col])
        node_boost = compute_boost(partition, pr, communities, node_scores, whitelist=wl_set)
        cat = map_boost_to_orders(cat, node_boost, dim)
    except Exception as e:
        print(f"  [{dim}] 图失败: {e}")
        cat[f'graph_{dim}_boost'] = 0

cat = compute_final_score(cat)

total = len(cat)
n_anom = (cat['risk_level'] == '异常').sum()
print(f"异常: {n_anom} 行 ({n_anom/total*100:.1f}%)", flush=True)
print("主类型分布:", cat[cat['risk_level']=='异常']['anomaly_type'].value_counts().head(8).to_dict())
print("Prophet命中:", cat['price_type'].fillna('').str.contains('Prophet').sum(), "行 | 高价聚量:", cat['concen_type'].fillna('').str.contains('高价聚量').sum(), "行")

out_dir = os.path.join(BASE, '类别专项检测', CAT_NAME)
os.makedirs(out_dir, exist_ok=True)
ts = int(time.time())
out_path = os.path.join(out_dir, f'{CAT_NAME}_全量检测_{ts}.xlsx')
cols = [c for c in ['order_id','ord_item_id','submit_time','smtr_name','thd_dept_short_name','sec_dept_short_name','sup_short_name',
        'consigner_name','rec_address','sku_name','proj_name','four_cat_name','tax_price','pur_qty','sub_ttl',
        'anomaly_type','risk_level',
        'price_type','qty_type','split_type','concen_type'] if c in cat.columns]
cat[cols].to_excel(out_path, index=False)
print(f"输出: {out_path} (耗时 {time.time()-t0:.0f}s)")
