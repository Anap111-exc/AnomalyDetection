# -*- coding: utf-8 -*-
"""图算法提权：分维度计算 boost 并映射回订单"""

import numpy as np
import pandas as pd
from config import *


def compute_boost(partition, pr, communities, node_scores, w1=GRAPH_W1, w2=GRAPH_W2,
                  whitelist=None):
    """
    计算每个节点的 graph_boost
    node_scores: {node_str: dim_score} 来自检测器原始分(boost前)
    公式: w1 * norm_PR + w2 * log(社区规模) * 社区原始均值分
    """
    if whitelist is None:
        whitelist = set()

    if not pr:
        return {}

    max_pr = max(pr.values()) if pr else 1
    node_boost = {}

    for node, cid in partition.items():
        if node in whitelist:
            node_boost[node] = 0.0
            continue

        # 仅对本身有风险的节点提权（原始分 >= 30 才参与）
        node_score = node_scores.get(node, 0)
        if node_score < 30:
            node_boost[node] = 0.0
            continue

        norm_pr = (pr.get(node, 0) / max_pr) * 100 if max_pr > 0 else 0
        comm_nodes = communities.get(cid, [node])
        comm_size = len(comm_nodes)

        # 社区原始均值分（boost前，无循环依赖）
        comm_scores = [node_scores.get(n, 0) for n in comm_nodes]
        comm_mean = np.mean(comm_scores) if comm_scores else 0

        boost = w1 * norm_pr + w2 * np.log(max(comm_size, 2)) * min(comm_mean, 30)
        node_boost[node] = min(round(boost, 2), 50)

    return node_boost


def map_boost_to_orders(df, node_boost, dim_name):
    """将节点boost映射到每行订单，只对原始分>0的行生效"""
    col_name = f'graph_{dim_name}_boost'
    score_col = f'{dim_name}_score'

    def row_boost(row):
        if row.get(score_col, 0) <= 0:
            return 0
        b = node_boost.get(str(row.get(COL_SMTR_NAME, '')), 0)
        s = node_boost.get(str(row.get(COL_SUP_NAME, '')), 0)
        sk = node_boost.get(str(row.get(COL_SKU_NAME, '')), 0)
        return max(b, s, sk)

    df[col_name] = df.apply(row_boost, axis=1)
    return df
