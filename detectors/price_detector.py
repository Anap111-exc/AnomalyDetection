# -*- coding: utf-8 -*-
"""价格异常检测器：规则(P1-P4) + per-SKU KDE核密度估计"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from config import *


def price_detector(df, rule_engine):
    print("  [price_detector] 执行中...")

    if 'price_score' not in df.columns:
        df['price_score'] = 0.0

    # Step 1: 规则引擎
    df = rule_engine.execute(df, dim='price')

    # Step 2: per-SKU KDE
    rule_hit = df['price_score'] >= 80
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

        prices = grp[COL_TAX_PRICE].values.reshape(-1, 1)
        try:
            kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(prices)
            log_density = kde.score_samples(prices)
            from scipy import stats
            ranks = stats.rankdata(-log_density)
            pct = ranks / len(ranks) * 100
            scores = np.where(pct >= 95, pct, 0)
            # 业务底线：价格必须偏离中位价至少50%
            median_price = np.median(prices)
            trivial_mask = grp[COL_TAX_PRICE] < median_price * 1.5
            scores[trivial_mask.values] = 0
        except Exception:
            scores = np.zeros(len(grp))

        grp['price_score_ml'] = scores
        df_list.append(grp)

    df_ml = pd.concat(df_list)
    if 'price_score_ml' in df_ml.columns:
        df.loc[df_ml.index, 'price_score_ml'] = df_ml['price_score_ml']
    if 'price_score_ml' not in df.columns:
        df['price_score_ml'] = 0.0
    df['price_score_ml'] = df['price_score_ml'].fillna(0)
    df['price_score'] = np.maximum(df['price_score'], df['price_score_ml'].fillna(0))

    print(f"    KDE处理 {len(df_ml)} 行, 规则命中 {(rule_hit).sum()} 行")
    return df
