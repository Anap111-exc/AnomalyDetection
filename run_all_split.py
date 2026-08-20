# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, os, time, warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_process import data_process, feature_engineer
from rule_engine import RuleEngine, register_all_rules
from detectors.split_detector import split_detector
from whitelist import load_whitelist, apply_whitelist

NEW_FILE = r'C:\Users\Administrator\Desktop\资料\内采异常采购分析\资料1\03新增字段数据\内采订单商品信息明细数据（20260101-0622）.xlsx'
OUT_DIR = r'C:\Users\Administrator\Desktop\python代码\虚拟环境1-测试\.venv\代码和结果6\类别专项检测'

df_all = pd.read_excel(NEW_FILE)
df = data_process(df_all)
df = feature_engineer(df)
print(f'清洗后: {len(df)} 行')

engine = RuleEngine(); register_all_rules(engine)
wl, wl_set = load_whitelist()
df = apply_whitelist(df, wl, wl_set)
df = split_detector(df, engine)

ts = int(time.time())
out_cols = ['order_id', 'ord_item_id', 'submit_time', 'smtr_name', 'thd_dept_short_name',
            'sup_short_name', 'consigner_name', 'rec_address', 'sku_name', 'proj_name',
            'four_cat_name', 'material_type_name', 'tax_price', 'pur_qty', 'sub_ttl',
            'split_type', 'split_rule_reason']
out_cols = [c for c in out_cols if c in df.columns]

out_path = os.path.join(OUT_DIR, f'全品类全量拆单检测_{ts}.xlsx')
df[out_cols].to_excel(out_path, index=False)

total = len(df)
hit = (df['split_score'] > 0).sum()
print(f'\n全量 {total} 行, 拆单 {hit} 行 ({(hit/total*100):.1f}%)')
print(f'结果: {out_path}')
