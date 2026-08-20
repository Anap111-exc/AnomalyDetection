# -*- coding: utf-8 -*-
"""图构建模块：为每个维度构建子图（白名单节点排除）"""

import pandas as pd
import numpy as np
import networkx as nx
from config import *


def build_dim_graph(df, dim_name, rolling_days=GRAPH_ROLLING_DAYS, whitelist=None):
    """
    构建单维度子图
    节点: 采购人/供应商/SKU（白名单节点不加入图）
    边: PURCHASE(采购人→供应商) + SUPPLY(供应商→SKU)
    """
    if whitelist is None:
        whitelist = set()

    if COL_SUBMIT_TIME not in df.columns:
        return nx.Graph()

    max_date = df[COL_SUBMIT_TIME].max()
    start_date = max_date - pd.Timedelta(days=rolling_days)
    df_window = df[(df[COL_SUBMIT_TIME] >= start_date) & (df[COL_SUBMIT_TIME] <= max_date)]

    if len(df_window) == 0:
        return nx.Graph()

    # 白名单过滤
    wl_str = set(str(e) for e in whitelist)

    weight_map = {
        'price': COL_SUB_TTL,
        'qty': COL_PUR_QTY,
        'concen': None,
        'ts': COL_SUB_TTL,
        'invis': COL_SUB_TTL,
    }
    weight_col = weight_map.get(dim_name, COL_SUB_TTL)

    G = nx.Graph()

    # 加节点（跳过白名单）
    for buyer in df_window[COL_SMTR_NAME].dropna().unique():
        if str(buyer) not in wl_str:
            G.add_node(str(buyer), type='buyer',
                       dept=str(df_window.loc[df_window[COL_SMTR_NAME] == buyer, COL_DEPT].iloc[0]) if COL_DEPT in df_window.columns else '')
    for sup in df_window[COL_SUP_NAME].dropna().unique():
        if str(sup) not in wl_str:
            G.add_node(str(sup), type='supplier',
                       company=str(df_window.loc[df_window[COL_SUP_NAME] == sup, COL_COMPANY].iloc[0]) if COL_COMPANY in df_window.columns else '')
    for sku in df_window[COL_SKU_NAME].dropna().unique():
        if str(sku) not in wl_str:
            G.add_node(str(sku), type='product',
                       category=str(df_window.loc[df_window[COL_SKU_NAME] == sku, COL_FOUR_CAT].iloc[0]) if COL_FOUR_CAT in df_window.columns else '')

    # PURCHASE 边
    if dim_name == 'concen':
        edges = df_window.groupby([COL_SMTR_NAME, COL_SUP_NAME])[COL_ORDER_ID].nunique()
    else:
        edges = df_window.groupby([COL_SMTR_NAME, COL_SUP_NAME])[weight_col].sum()

    for (buyer, sup), w in edges.items():
        if w > 0 and str(buyer) not in wl_str and str(sup) not in wl_str:
            G.add_edge(str(buyer), str(sup), weight=float(w), relation='PURCHASE')

    # SUPPLY 边
    if dim_name == 'concen':
        supply_edges = df_window.groupby([COL_SUP_NAME, COL_SKU_NAME])[COL_ORDER_ID].nunique()
    else:
        supply_edges = df_window.groupby([COL_SUP_NAME, COL_SKU_NAME])[weight_col].sum()

    for (sup, sku), w in supply_edges.items():
        if w > 0 and str(sup) not in wl_str and str(sku) not in wl_str:
            G.add_edge(str(sup), str(sku), weight=float(w), relation='SUPPLY')

    return G
