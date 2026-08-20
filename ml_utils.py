# -*- coding: utf-8 -*-
"""ML 通用工具"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_anomaly_scores(raw, low=1, high=100):
    """MinMax归一化，处理全相同值的退化情况"""
    raw = np.array(raw, dtype=float).reshape(-1, 1)
    if raw.max() - raw.min() < 1e-8:
        return np.full(raw.shape[0], (low + high) / 2)
    scaler = MinMaxScaler(feature_range=(low, high))
    return scaler.fit_transform(raw).flatten()
