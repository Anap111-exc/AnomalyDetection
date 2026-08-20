# -*- coding: utf-8 -*-
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

KW = '支架'
raw = glob.glob(os.path.join(BASE, "内采订单商品信息明细数据*.xlsx"))[0]
df_all = data_process(pd.read_excel(raw)); df_all = feature_engineer(df_all)
wl, wl_set = load_whitelist()
cat = df_all[df_all[COL_FOUR_CAT].astype(str).str.contains(KW, na=False)].copy()
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
        if G.number_of_nodes() == 0: continue
        partition, communities = run_louvain(G)
        pr = run_pagerank(G)
        node_scores = {}
        score_col = f'{dim}_score'
        for _, row in cat.iterrows():
            for entity in [str(row.get(COL_SMTR_NAME, '')), str(row.get(COL_SUP_NAME, '')), str(row.get(COL_SKU_NAME, ''))]:
                if entity: node_scores[entity] = max(node_scores.get(entity, 0), row[score_col])
        node_boost = compute_boost(partition, pr, communities, node_scores, whitelist=wl_set)
        cat = map_boost_to_orders(cat, node_boost, dim)
    except Exception as e:
        print('图失败', dim, e)
        cat[f'graph_{dim}_boost'] = 0
cat = compute_final_score(cat)

n = (cat['risk_level'] == '异常').sum()
print(f'显示器支架: {len(cat)}行 异常{n} ({round(100*n/len(cat),1)}%)')
print('anomaly_type 分布:')
print(cat[cat['risk_level']=='异常']['anomaly_type'].value_counts().head(8).to_string())
print()
print('数量相关行 qty_type:')
q = cat[cat['qty_type'].fillna('') != '']
print(q['qty_type'].value_counts().head(5).to_string())
print()
print('样例(数量异常行):')
for _, r in q.head(4).iterrows():
    ts = r.submit_time.strftime('%m-%d')
    print(f'  {ts} {str(r.sku_name)[:16]:<18} 量{r.pur_qty:>4} | anomaly_type={r.anomaly_type} | qty_type={r.qty_type}')

ts = int(time.time())
out_dir = os.path.join(BASE, '类别专项检测', '显示器支架')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'显示器支架_全量检测_{ts}.xlsx')
cols = [c for c in ['order_id','ord_item_id','submit_time','smtr_name','thd_dept_short_name','sec_dept_short_name','sup_short_name',
        'consigner_name','rec_address','sku_name','proj_name','four_cat_name','tax_price','pur_qty','sub_ttl',
        'anomaly_type','risk_level','price_type','qty_type','split_type','concen_type'] if c in cat.columns]
cat[cols].to_excel(out_path, index=False)
print(f'\n输出: {out_path}')