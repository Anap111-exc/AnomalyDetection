# -*- coding: utf-8 -*-
"""数量异常检测器：规则(Q1/Q3) + per-SKU KDE CDF分位数
KDE拟合用SKU全样本（含规则命中行，分布不失真），打分覆盖全部行（规则命中行取 max(规则分, KDE分)）"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from config import *


def qty_detector(df, rule_engine):
    print(f"  [qty_detector] 执行中...", flush=True)

    if 'qty_score' not in df.columns:
        df['qty_score'] = 0.0

    # Step 1: 规则引擎
    df = rule_engine.execute(df, dim='qty')

    # Step 2: per-SKU KDE CDF分位数
    large_sample = df['sku_sample_count'] >= MIN_SKU_SAMPLES

    # 拟合用全样本（避免排除大单后分布失真），打分覆盖全部行（规则命中行取两者最大值）
    fit_df = df[large_sample]
    if len(fit_df) == 0:
        print(f"    KDE跳过", flush=True)
        return df

    df_list = []
    for grp_key, grp in fit_df.groupby(COL_SKU_NAME):
        if len(grp) < 5:
            continue

        try:
            qty_vals = grp[COL_PUR_QTY].values.reshape(-1, 1)
            kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(qty_vals)

            # 在网格上计算概率密度，构造CDF
            qty_max = qty_vals.max()
            grid = np.linspace(0, max(qty_max * 1.5, 50), 2000).reshape(-1, 1)
            log_dens = kde.score_samples(grid)
            dens = np.exp(log_dens)
            cdf = np.cumsum(dens)
            cdf = cdf / cdf[-1]

            # 求KDE拟合分布的RULE_QTY_KDE_PERCENTILE分位数对应的数量值（默认95%，提高精确率）
            idx_p = np.searchsorted(cdf, RULE_QTY_KDE_PERCENTILE)
            threshold_kde = float(grid[idx_p][0])

            # 后置过滤：不低于中位数×2或10，防止阈值过低
            median_qty = np.median(qty_vals)
            threshold = max(threshold_kde, median_qty * 2, 10)

            # 数量 ≥ 阈值才给分，分数=超阈值比例映射到90~100
            raw_scores = np.where(qty_vals.flatten() >= threshold, qty_vals.flatten() / threshold * 100, 0)
            scores = np.minimum(raw_scores, 100).round(1)
        except Exception:
            scores = np.zeros(len(grp))

        grp['qty_score_ml'] = scores
        df_list.append(grp)

    if df_list:
        scored = pd.concat(df_list)
        # 回写：全部行（含规则命中行）均采用KDE分，最终取 max(规则分, KDE分)
        df['qty_score_ml'] = 0.0
        df.loc[scored.index, 'qty_score_ml'] = scored['qty_score_ml']

    if 'qty_score_ml' not in df.columns:
        df['qty_score_ml'] = 0.0
    df['qty_score_ml'] = df['qty_score_ml'].fillna(0)

    df['qty_score'] = np.maximum(df['qty_score'], df['qty_score_ml'].fillna(0))
    # 标注来源，让结果显示时能区分 KDE 与规则
    kde_mask = df['qty_score_ml'] > 0
    if 'qty_rule_reason' not in df.columns:
        df['qty_rule_reason'] = ''
    df.loc[kde_mask, 'qty_rule_reason'] = \
        df.loc[kde_mask, 'qty_rule_reason'].fillna('') + ';KDE:数量分布离群'

    print(f"    KDE处理 {len(fit_df)} 行", flush=True)
    return df
