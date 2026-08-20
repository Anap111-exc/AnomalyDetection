# -*- coding: utf-8 -*-
"""白名单管理"""

import os
import pandas as pd
from config import *


def load_whitelist(filepath=None):
    """加载白名单，返回 dict 和 set"""
    whitelist = {}
    wl_set = set()
    path = filepath or WHITELIST_FILE
    if not os.path.exists(path):
        return whitelist, wl_set

    df_wl = pd.read_excel(path)
    for _, row in df_wl.iterrows():
        entity_id = str(row.get('entity_id', ''))
        if entity_id:
            whitelist[entity_id] = {
                'reason': row.get('reason', ''),
                'expiry': pd.to_datetime(row.get('expiry_date', '2099-12-31')),
                'approved_by': row.get('approved_by', ''),
            }
            wl_set.add(entity_id)
    return whitelist, wl_set


def apply_whitelist(df, whitelist, wl_set):
    """给 df 打白名单标签"""
    if not wl_set:
        df['is_whitelisted'] = 0
        df['whitelist_reason'] = ''
        return df

    wl_entities = set(str(e) for e in wl_set)

    df['is_whitelisted'] = (
        df[COL_SMTR_NAME].astype(str).isin(wl_entities) |
        df[COL_SUP_NAME].astype(str).isin(wl_entities) |
        df[COL_SKU_NAME].astype(str).isin(wl_entities)
    ).astype(int)

    df['whitelist_reason'] = ''
    for idx, row in df.iterrows():
        reasons = []
        for entity in [str(row[COL_SMTR_NAME]), str(row[COL_SUP_NAME]), str(row[COL_SKU_NAME])]:
            if entity in whitelist and pd.Timestamp.now() <= whitelist[entity]['expiry']:
                reasons.append(f"{entity}:{whitelist[entity]['reason']}")
        df.at[idx, 'whitelist_reason'] = ';'.join(reasons)

    return df
