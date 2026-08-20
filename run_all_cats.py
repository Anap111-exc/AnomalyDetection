# -*- coding: utf-8 -*-
"""批量品类全检测（新目录）"""
import pandas as pd, numpy as np, os, time, warnings, glob, sys, io
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
BASE = r"C:\Users\Administrator\Desktop\python代码\虚拟环境1-测试\代码和结果6"
sys.path.insert(0, BASE); os.chdir(BASE)
import logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

from config import *
from data_process import data_process, feature_engineer
from rule_engine import RuleEngine, register_all_rules
from detectors import price_detector, qty_detector, split_detector, concen_detector
from graph.build_graph import build_dim_graph
from graph.graph_algo import run_louvain, run_pagerank
from graph.boost import compute_boost, map_boost_to_orders
from fusion import compute_final_score
from whitelist import load_whitelist, apply_whitelist

CATEGORIES = [
    ('标签机', 'contains', '标签机'),
    ('茶叶', 'exact', '茶叶'),
    ('打印机及复印机配件', 'exact', '打印机及复印机配件'),
    ('柜式空调-挂式空调', 'exact', '家用空调'),
    ('牛奶（不含鲜奶）', 'exact', '牛奶（不含鲜奶）'),
    ('排插插排', 'contains', '排插'),
    ('显示器支架', 'contains', '支架'),
]

raw = glob.glob(os.path.join(BASE, "内采订单商品信息明细数据*.xlsx"))[0]
print("清洗+特征工程...", flush=True)
df_all = data_process(pd.read_excel(raw))
df_all = feature_engineer(df_all)
wl, wl_set = load_whitelist()

OUT_ROOT = os.path.join(BASE, '类别专项检测')
for cat_name, mode, kw in CATEGORIES:
    t0 = time.time()
    print(f"\n{'='*60}\n品类: {cat_name}\n{'='*60}", flush=True)
    if mode == 'exact':
        cat = df_all[df_all[COL_FOUR_CAT] == kw].copy()
    else:
        cat = df_all[df_all[COL_FOUR_CAT].astype(str).str.contains(kw, na=False)].copy()
    print(f"子集: {len(cat)} 行, {cat['sku_name'].nunique()} SKU", flush=True)
    if len(cat) == 0:
        print("  空子集，跳过"); continue

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
    print(f"  异常: {n_anom} 行 ({n_anom/total*100:.1f}%)", flush=True)
    print("  主类型:", cat[cat['risk_level']=='异常']['anomaly_type'].value_counts().head(6).to_dict())

    out_dir = os.path.join(OUT_ROOT, cat_name)
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    out_path = os.path.join(out_dir, f'{cat_name}_全量检测_{ts}.xlsx')
    cols = [c for c in ['order_id','ord_item_id','submit_time','smtr_name','thd_dept_short_name','sec_dept_short_name','sup_short_name',
            'consigner_name','rec_address','sku_name','proj_name','four_cat_name','tax_price','pur_qty','sub_ttl',
            'anomaly_type','risk_level',
            'price_type','qty_type','split_type','concen_type'] if c in cat.columns]
    cat[cols].to_excel(out_path, index=False)
    print(f"  输出: {out_path}  (耗时 {time.time()-t0:.0f}s)", flush=True)

print("\n全部完成!")