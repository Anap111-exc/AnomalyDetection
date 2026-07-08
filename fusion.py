# -*- coding: utf-8 -*-
"""简化融合：规则优先 → ML+图兜底"""

import numpy as np
import pandas as pd
from config import *

DIM_CN = {
    'price': '价格异常',
    'qty': '数量异常',
    'split': '拆单',
    'concen': '高价聚量',
    'ts': '时序异常',
    'invis': '隐形异常',
}


def compute_final_score(df):
    """规则引擎命中→直接异常；未命中→ML+图任一维度超阈值→异常"""

    # Step 1: 分维度 boosted_score (上限 100)
    for dim in DIMS:
        boost_col = f'graph_{dim}_boost'
        score_col = f'{dim}_score'
        if boost_col in df.columns:
            df[f'boosted_{dim}'] = (df[score_col].fillna(0) + df[boost_col].fillna(0)).clip(upper=100)
        else:
            df[f'boosted_{dim}'] = df[score_col].fillna(0).clip(upper=100)

    # Step 2: 判定
    df['is_violated'] = df.get('is_violated', 0).fillna(0)
    df['is_whitelisted'] = df.get('is_whitelisted', 0).fillna(0)

    risk_levels = []
    anomaly_types = []
    anomaly_details = []

    boosted_cols = [f'boosted_{d}' for d in DIMS]

    for _, row in df.iterrows():
        if row['is_whitelisted'] == 1:
            risk_levels.append('无异常')
            anomaly_types.append('')
            anomaly_details.append('白名单')
            continue

        if row['is_violated'] == 100:
            risk_levels.append('异常')
            types = [DIM_CN[d] for d in DIMS if row.get(f'{d}_score', 0) >= 80]
            atype = types[0] if types else '规则命中'
            if atype == '高价聚量':
                types.append('(注意该申购人近期同品类订单)')
            anomaly_types.append(atype)
            anomaly_details.append(','.join(types) if types else '规则命中')
            continue

        # ML+图判定（按维度使用不同阈值）
        max_dim = DIMS[0]
        max_val = 0
        for d in DIMS:
            v = row.get(f'boosted_{d}', 0)
            threshold = SPLIT_BINARY_THRESHOLD if d == 'split' else BINARY_THRESHOLD
            if v >= threshold and v > max_val:
                max_val = v
                max_dim = d

        if max_val > 0:
            risk_levels.append('异常')
            atype = DIM_CN[max_dim]
            # 高价聚量标签污染修复：只有规则命中的才标"高价聚量"
            if atype == '高价聚量':
                concen_reason = row.get('concen_rule_reason', '')
                if pd.isna(concen_reason) or concen_reason == '':
                    atype = '隐形异常'
            anomaly_types.append(atype)
            types = [DIM_CN[d] for d in DIMS if row.get(f'{d}_score', 0) > 40]
            if atype == '隐形异常' and '高价聚量' not in types:
                types.append('(行为模式异常)')
            # 高价聚量追加同品类提醒
            if atype == '高价聚量':
                types.append('(注意该申购人近期同品类订单)')
            anomaly_details.append(','.join(types) if types else atype)
        else:
            risk_levels.append('无异常')
            anomaly_types.append('')
            anomaly_details.append('')

    df['risk_level'] = risk_levels
    df['anomaly_type'] = anomaly_types
    df['anomaly_detail'] = anomaly_details

    # 清理
    for d in DIMS:
        df.drop(columns=[f'boosted_{d}'], inplace=True, errors='ignore')

    return df
