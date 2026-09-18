# -*- coding: utf-8 -*-
"""高价聚量检测器：规则(C2-C4) + Prophet+DBSCAN并行"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from config import *

# Prophet 可选
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

# Prophet+DBSCAN 参数
PROPHET_MIN_TRAIN = 10          # 最少训练点数
DBSCAN_EPS = 0.5                # 邻域半径（标准化后）
DBSCAN_MIN_SAMPLES = 3          # 核心点最小邻居数


def _prophet_price_anomaly(df):
    """per-SKU Prophet时间序列价格异常检测
    返回 (布尔异常列, 95%置信区间上界列)；无 Prophet 覆盖的行上界为 NaN"""
    if not HAS_PROPHET:
        return pd.Series(False, index=df.index), pd.Series(np.nan, index=df.index)

    result = pd.Series(False, index=df.index)
    ci = pd.Series(np.nan, index=df.index)          # 每行 yhat_upper（95% 置信区间上界）
    sku_counts = df.groupby(COL_SKU_NAME).size()
    eligible = sku_counts[sku_counts >= PROPHET_MIN_TRAIN].index

    if len(eligible) == 0:
        return result, ci

    for n_processed, sku in enumerate(eligible, 1):
        if n_processed % 200 == 0 or n_processed == len(eligible):
            print(f"      Prophet进度: {n_processed}/{len(eligible)} SKU", flush=True)
        mask = df[COL_SKU_NAME] == sku
        sub = df[mask].sort_values(COL_SUBMIT_TIME)
        try:
            daily = sub[[COL_SUBMIT_TIME, COL_TAX_PRICE]].rename(
                columns={COL_SUBMIT_TIME: 'ds', COL_TAX_PRICE: 'y'}
            )
            daily_unique = daily.drop_duplicates(subset='ds')
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05,
            )
            model.rng = np.random.default_rng(42)  # 固定区间采样种子，消除边界行运行间翻转
            model.fit(daily_unique)
            forecast = model.predict(daily[['ds']])
            # 仅判定高于预测上限的价格为异常，且须高于"该行之前的常态价"5%
            # （前向中位数=时间穿越基准：降价转折点前的旧价不判；首笔无历史回退全历史中位数）
            yhat = forecast['yhat'].values
            upper = forecast['yhat_upper'].values
            lower = forecast['yhat_lower'].values
            sub_sorted = sub.sort_values(COL_SUBMIT_TIME)
            prev_median = sub_sorted[COL_TAX_PRICE].expanding().median().shift(1)
            prev_median = prev_median.fillna(np.median(sub[COL_TAX_PRICE].values))
            median_price = prev_median.values
            anom = (daily['y'].values > upper) & \
                   (daily['y'].values > median_price * (1 + PROPHET_MIN_MEDIAN_DEV))
            # 段首过滤：同价连续段只保留段内前N笔（突变窗口内标，后续同价单为新常态不重复标）
            is_jump = (sub_sorted[COL_TAX_PRICE] != sub_sorted[COL_TAX_PRICE].shift(1)).fillna(True)
            seg_id = is_jump.cumsum()
            seg_rank = sub_sorted.groupby(seg_id).cumcount() + 1
            keep = seg_rank <= PROPHET_JUMP_KEEP_N
            anom = anom & keep.values
            result.loc[sub.index] = anom
            ci.loc[sub.index] = upper           # upper 与 sub 按时间排序后的行序一致
        except Exception:
            continue

    return result, ci


def _dbscan_quantity_anomaly(df):
    """per-SKU DBSCAN数量聚类检测，返回布尔列（只标记噪声点中的高量）"""
    result = pd.Series(False, index=df.index)

    for sku, grp in df.groupby(COL_SKU_NAME):
        n = len(grp)
        if n < 2:
            continue
        values = grp[COL_PUR_QTY].values

        if n >= 3:
            try:
                scaled = StandardScaler().fit_transform(np.log1p(values).reshape(-1, 1))
                labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(scaled)
                clusters = set(labels) - {-1}
                if clusters:
                    # 噪声点(-1)中超过簇中位数3倍的标记
                    cluster_vals = values[labels != -1]
                    cluster_median = np.median(cluster_vals)
                    noise_mask = labels == -1
                    if noise_mask.any() and cluster_median > 0:
                        noise_vals = values[noise_mask]
                        result.loc[grp.index[noise_mask]] = noise_vals > cluster_median * 2
                    continue
            except Exception:
                pass

        # 回退规则：n<3或DBSCAN未分出簇时，超过中位数2倍标记
        median_qty = np.median(values)
        if median_qty > 0:
            result.loc[grp.index[values > median_qty * 2]] = True

    return result


def _inverted_v_mask(df, cand_mask):
    """倒V型波动过滤（领导口径）：候选行须在其后'未来14天'或'未来5单'内
    存在价格回落到 该笔价格×0.95 以下的订单，任一窗口满足即保留。
    只做离线判定（用到未来数据），返回布尔列。"""
    result = pd.Series(False, index=df.index)
    if cand_mask.sum() == 0:
        return result
    NS_DAY = 86_400_000_000_000
    for sku in df.loc[cand_mask, COL_SKU_NAME].unique():
        sub = df[df[COL_SKU_NAME] == sku].sort_values(COL_SUBMIT_TIME)
        if len(sub) < 2:
            continue
        times = pd.to_datetime(sub[COL_SUBMIT_TIME], utc=True).dt.tz_convert(None).astype('datetime64[ns]').astype('int64').values
        prices = sub[COL_TAX_PRICE].values
        idx_arr = list(sub.index)
        for pos, i in enumerate(idx_arr):
            if not cand_mask.loc[i]:
                continue
            line = prices[pos] * RULE_CONCEN_INVERTED_V_DROP
            ok = False
            # 未来5单窗口
            for j in range(pos + 1, min(pos + 1 + RULE_CONCEN_INVERTED_V_NEXT_N, len(sub))):
                if prices[j] <= line:
                    ok = True
                    break
            # 未来14天窗口（时间单调递增，超窗即断）
            if not ok:
                for j in range(pos + 1, len(sub)):
                    if times[j] - times[pos] > RULE_CONCEN_INVERTED_V_DAYS * NS_DAY:
                        break
                    if prices[j] <= line:
                        ok = True
                        break
            if ok:
                result.loc[i] = True
    return result


def _detect_prophet_dbscan(df):
    """Prophet+DBSCAN并行检测
    返回 (concen_scores, price_anom_scores, price_anom, is_high_price_agg, price_ci)"""
    df = df.copy()
    print(f"    [并行] Prophet时序价格异常 + DBSCAN数量聚类...", flush=True)

    if USE_PROPHET:
        price_anom, price_ci = _prophet_price_anomaly(df)
    else:
        price_anom, price_ci = (pd.Series(False, index=df.index),
                                pd.Series(np.nan, index=df.index))
    qty_anom = _dbscan_quantity_anomaly(df)

    print(f"      Prophet价格异常: {price_anom.sum()} 行", flush=True)
    print(f"      DBSCAN数量异常: {qty_anom.sum()} 行", flush=True)

    # 数量异常信号 = 数量检测器(qty_score>=80) ∪ DBSCAN聚类
    # 双口径：qty_detector的KDE CDF主判 + DBSCAN聚类兜底（两链互相补漏）
    if 'qty_score' in df.columns:
        qty_signal = (df['qty_score'].fillna(0).values >= 80) | qty_anom.values
    else:
        qty_signal = qty_anom.values

    # 价格异常信号 = Prophet时序判定 ∪ 价格检测器规则/KDE分数(price_score>=80)
    price_signal = price_anom.values | (df['price_score'].fillna(0).values >= 80)

    # 高价聚量 = 价格异常 AND 数量异常
    is_high_price_agg = price_signal & qty_signal
    print(f"      高价聚量(交集): {is_high_price_agg.sum()} 行", flush=True)

    # 倒V型过滤：交集聚量须满足'先涨后回落'（未来14天或未来5单内价格回落≥5%）
    keep_agg = _inverted_v_mask(df, pd.Series(is_high_price_agg, index=df.index))
    is_high_price_agg = is_high_price_agg & keep_agg.values
    print(f"      倒V过滤后交集聚量: {is_high_price_agg.sum()} 行", flush=True)

    concen_scores = np.zeros(len(df))
    concen_scores[is_high_price_agg] = 100

    # 所有 Prophet 价格异常行回写 price_score（含高价聚量交集行，价格异常信息不丢失）
    # 90分 = BINARY_THRESHOLD，确保 Prophet 时序异常在融合层被判定为异常
    price_scores = np.zeros(len(df))
    price_scores[price_anom.values] = 90

    return concen_scores, price_scores, price_anom, is_high_price_agg, price_ci


def concen_detector(df, rule_engine):
    print(f"  [concen_detector] 执行中...", flush=True)

    if 'concen_score' not in df.columns:
        df['concen_score'] = 0.0

    # Step 1: 规则引擎
    df = rule_engine.execute(df, dim='concen')

    # Step 2: Prophet+DBSCAN 并行检测
    if COL_SUBMIT_TIME in df.columns:
        concen_scores, price_anom_scores, price_anom, is_agg, price_ci = _detect_prophet_dbscan(df)
        prophet_mask = concen_scores > 0
        # 置信区间上界（Prophet yhat_upper）与偏离率=(单价-上界)/上界；无 Prophet 覆盖的行留 NaN
        df['confidence_interval'] = price_ci.values
        with np.errstate(divide='ignore', invalid='ignore'):
            df['deviation'] = (df[COL_TAX_PRICE] - df['confidence_interval']) / df['confidence_interval']
        df.loc[~np.isfinite(df['deviation']), 'deviation'] = np.nan
        # C3/C4 规则命中的聚量同样要求倒V型波动（C2 高价供应商依赖除外）
        if 'concen_rule_reason' in df.columns:
            c34_mask = df['concen_rule_reason'].fillna('').astype(str).str.contains('C3:|C4:', na=False)
            if c34_mask.any():
                keep_c34 = _inverted_v_mask(df, c34_mask)
                bad_c34 = c34_mask & ~keep_c34
                df.loc[bad_c34, 'concen_score'] = 0.0
                print(f"      C3/C4规则聚量倒V过滤: {c34_mask.sum()} 行 -> 保留 {keep_c34.sum()} 行", flush=True)
        df['concen_score'] = np.maximum(df['concen_score'], concen_scores)
        # 将 Prophet 价格异常回写到 price_score（含高价聚量行）
        df['price_score'] = np.maximum(df['price_score'], price_anom_scores)
        # 价格异常来源标注，让结果显示时能区分 Prophet 与规则/KDE
        if 'price_rule_reason' not in df.columns:
            df['price_rule_reason'] = ''
        pp_mask = pd.Series(price_anom, index=df.index)
        df.loc[pp_mask, 'price_rule_reason'] = \
            df.loc[pp_mask, 'price_rule_reason'].fillna('') + ';Prophet:时序价格异常'
        # 设置规则原因，让融合层识别为"高价聚量"而非改为"隐形异常"
        if 'concen_rule_reason' not in df.columns:
            df['concen_rule_reason'] = ''
        df.loc[prophet_mask, 'concen_rule_reason'] = df.loc[prophet_mask, 'concen_rule_reason'].fillna('') + ';Prophet:时序价格异常+数量异常'

    print(f"    规则命中 {(df['concen_score'] >= 70).sum()} 行", flush=True)
    return df
