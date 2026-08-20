# -*- coding: utf-8 -*-
"""图算法：Louvain社区发现 + PageRank中心性"""

import networkx as nx

try:
    import community.community_louvain as cl
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


def run_louvain(G):
    """Louvain社区发现"""
    if not HAS_LOUVAIN or G.number_of_nodes() == 0:
        return {}, {}
    try:
        partition = cl.best_partition(G, weight='weight', random_state=42)
    except Exception:
        partition = {n: 0 for n in G.nodes()}

    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)
    return partition, communities


def run_pagerank(G):
    """PageRank中心性（在原始无向图上算，用 weight 属性）"""
    if G.number_of_nodes() == 0:
        return {}
    try:
        pr = nx.pagerank(G, weight='weight', alpha=0.85)
    except Exception:
        pr = {n: 1.0 / max(G.number_of_nodes(), 1) for n in G.nodes()}
    return pr
