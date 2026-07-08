# -*- coding: utf-8 -*-
"""数量异常检测器：规则(Q1-Q2) + per-SKU KDE核密度估计"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from config import *


def qty_detector(df, rule_engine):
    print("  [qty_detector] 执行中...")

    if 'qty_score' not in df.columns:
        df['qty_score'] = 0.0

    # Step 1: 规则引擎
    df = rule_engine.execute(df, dim='qty')

    # Step 2: per-SKU KDE
    rule_hit = df['qty_score'] >= 70
    large_sample = df['sku_sample_count'] >= MIN_SKU_SAMPLES
    df_ml = df[large_sample & ~rule_hit].copy()

    if len(df_ml) == 0:
        print(f"    KDE跳过")
        return df

    df_list = []
    for grp_key, grp in df_ml.groupby(COL_SKU_NAME):
        if len(grp) < 5:
            df_list.append(grp)
            continue

        qty_vals = grp[COL_PUR_QTY].values.reshape(-1, 1)
        try:
            kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(qty_vals)
            log_density = kde.score_samples(qty_vals)
            from scipy import stats
            ranks = stats.rankdata(-log_density)
            pct = ranks / len(ranks) * 100
            scores = np.where(pct >= 95, pct, 0)
        except Exception:
            scores = np.zeros(len(grp))

        grp['qty_score_ml'] = scores
        df_list.append(grp)

    df_ml = pd.concat(df_list)
    if 'qty_score_ml' in df_ml.columns:
        df.loc[df_ml.index, 'qty_score_ml'] = df_ml['qty_score_ml']
    if 'qty_score_ml' not in df.columns:
        df['qty_score_ml'] = 0.0
    df['qty_score_ml'] = df['qty_score_ml'].fillna(0)

    df['qty_score'] = np.maximum(df['qty_score'], df['qty_score_ml'].fillna(0))

    # 数量异常后过滤（使用全局均量）
    sku_global_avg = df.groupby(COL_SKU_NAME)[COL_PUR_QTY].transform('mean')
    qty_threshold = np.maximum(10, sku_global_avg * 3)
    trivial_mask = (df[COL_PUR_QTY] < qty_threshold)
    df.loc[trivial_mask, 'qty_score'] = 0

    print(f"    KDE处理 {len(df_ml)} 行")
    return df
