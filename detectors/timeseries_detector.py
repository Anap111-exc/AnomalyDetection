# -*- coding: utf-8 -*-
"""时序突变检测器：规则(T1-T2) + PELT变点检测"""

import pandas as pd
import numpy as np
from config import *

# PELT 可选依赖
try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False


def detect_changepoints(sequence, pen=3):
    """PELT变点检测，返回突变点索引列表"""
    if not HAS_RUPTURES or len(sequence) < 5:
        return []
    try:
        algo = rpt.Pelt(model="rbf").fit(np.array(sequence, dtype=float).reshape(-1, 1))
        cp = algo.predict(pen=pen)
        return cp[:-1]  # 去掉末尾
    except Exception:
        return []


def ts_detector(df, rule_engine):
    print("  [ts_detector] 执行中...")

    if 'ts_score' not in df.columns:
        df['ts_score'] = 0.0

    # Step 1: 规则引擎
    df = rule_engine.execute(df, dim='ts')

    if not HAS_RUPTURES:
        print("    ruptures未安装，PELT跳过")
        return df

    # Step 2: PELT变点检测 — 采购人×月
    if COL_SMTR_NAME in df.columns:
        buyer_monthly = df.groupby([COL_SMTR_NAME, 'submit_month']).size().reset_index(name='_cnt')
        buyer_monthly = buyer_monthly.sort_values([COL_SMTR_NAME, 'submit_month'])

        scores = {}
        for buyer, grp in buyer_monthly.groupby(COL_SMTR_NAME):
            seq = grp['_cnt'].values
            cp = detect_changepoints(seq, pen=3)
            for idx in cp:
                if idx < len(grp):
                    period = grp.iloc[idx]['submit_month']
                    scores[(buyer, period)] = 85

        if scores:
            def get_buyer_score(r):
                return scores.get((r[COL_SMTR_NAME], r['submit_month']), 0)
            df['ts_score_pelt'] = df.apply(get_buyer_score, axis=1)
            df['ts_score_ml'] = df['ts_score_pelt'].fillna(0)
            df['ts_score'] = np.maximum(df['ts_score'], df['ts_score_ml'].fillna(0))

    # Step 3: PELT — SKU×月
    if COL_SKU_ID in df.columns:
        sku_monthly = df.groupby([COL_SKU_ID, 'submit_month'])[COL_TAX_PRICE].mean().reset_index()
        sku_monthly = sku_monthly.sort_values([COL_SKU_ID, 'submit_month'])

        sku_scores = {}
        for sku, grp in sku_monthly.groupby(COL_SKU_ID):
            seq = grp[COL_TAX_PRICE].values
            cp = detect_changepoints(seq, pen=4)
            for idx in cp:
                if idx < len(grp):
                    period = grp.iloc[idx]['submit_month']
                    sku_scores[(sku, period)] = 80

        if sku_scores:
            def get_sku_score(r):
                return sku_scores.get((r[COL_SKU_ID], r['submit_month']), 0)
            df['ts_score_sku'] = df.apply(get_sku_score, axis=1)
            df['ts_score'] = np.maximum(df['ts_score'], df['ts_score_sku'].fillna(0))

    # Step 4: PELT — 供应商×周
    if COL_SUP_NAME in df.columns and 'submit_week' in df.columns:
        sup_weekly = df.groupby([COL_SUP_NAME, 'submit_week']).size().reset_index(name='_cnt')
        sup_weekly = sup_weekly.sort_values([COL_SUP_NAME, 'submit_week'])

        sup_scores = {}
        for sup, grp in sup_weekly.groupby(COL_SUP_NAME):
            seq = grp['_cnt'].values
            cp = detect_changepoints(seq, pen=3)
            for idx in cp:
                if idx < len(grp):
                    period = grp.iloc[idx]['submit_week']
                    sup_scores[(sup, period)] = 75

        if sup_scores:
            def get_sup_score(r):
                return sup_scores.get((r[COL_SUP_NAME], r['submit_week']), 0)
            df['ts_score_sup'] = df.apply(get_sup_score, axis=1)
            df['ts_score'] = np.maximum(df['ts_score'], df['ts_score_sup'].fillna(0))

    print(f"    PELT完成")
    return df
