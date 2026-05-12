# -*- coding: utf-8 -*-
"""
动态阈值单调性检查与修正模块

由于环境波动，原始预测的 Y_hat(t) 可能局部下降，
导致动态阈值 T_dyn(t) 局部上升，不符合剩磁单调衰减的物理规律。
本模块检查局部上升并生成单调修正版。
"""

import pandas as pd
import numpy as np
from .utils import setup_logger, print_section_header, get_output_dir, get_static_threshold, SAMPLE_TYPES

logger = setup_logger()


def check_and_fix_monotonicity(df_original_thresholds):
    """
    检查动态阈值单调性并对局部上升进行修正

    单调修正: Y_mono(t) = max(Y_hat(1), ..., Y_hat(t))
              T_dyn_mono(t) = T0 * exp[-Y_mono(t)]

    Parameters
    ----------
    df_original_thresholds : pd.DataFrame
        原始动态阈值表，列包含 'day' 和 '{sample_type}_阈值'

    Returns
    -------
    df_check : pd.DataFrame
        单调性检查结果
    df_monotonic : pd.DataFrame
        单调修正后的动态阈值表
    """
    print_section_header("动态阈值单调性检查与修正")

    df_th = df_original_thresholds.copy()
    check_rows = []
    monotonic_days = df_th['day'].values

    df_mono = pd.DataFrame({'day': monotonic_days})

    has_violation = False

    for sample_type in SAMPLE_TYPES:
        col = f'{sample_type}_阈值'
        if col not in df_th.columns:
            continue

        T_vals = df_th[col].values
        T_static = get_static_threshold(sample_type)
        Y_hat = -np.log(np.maximum(T_vals, 1e-10) / T_static)

        Y_mono = np.maximum.accumulate(Y_hat)
        T_mono = T_static * np.exp(-Y_mono)

        for t_idx in range(1, len(T_vals)):
            if T_vals[t_idx] > T_vals[t_idx - 1] + 1e-6:
                has_violation = True
                day = monotonic_days[t_idx]
                check_rows.append({
                    '样品类型': sample_type,
                    '天数': day,
                    '原始阈值(mT)': round(T_vals[t_idx], 6),
                    '前一天阈值(mT)': round(T_vals[t_idx - 1], 6),
                    '变化量(mT)': round(T_vals[t_idx] - T_vals[t_idx - 1], 6),
                    '修正后阈值(mT)': round(T_mono[t_idx], 6),
                })

        df_mono[col] = T_mono

    df_check = pd.DataFrame(check_rows)
    if has_violation:
        logger.warning(f"发现{len(df_check)}处阈值局部上升（{df_check['样品类型'].nunique()}种样品类型）")
    else:
        logger.info("未发现阈值局部上升，单调性满足")

    output_dir = get_output_dir()
    if has_violation:
        df_check.to_excel(output_dir / 'dynamic_threshold_monotonic_check.xlsx', index=False, engine='openpyxl')
        logger.info(f"单调性检查结果已保存: {output_dir / 'dynamic_threshold_monotonic_check.xlsx'}")

    df_mono.to_excel(output_dir / 'monotonic_dynamic_threshold_1_90.xlsx', index=False, engine='openpyxl')
    logger.info(f"单调修正阈值已保存: {output_dir / 'monotonic_dynamic_threshold_1_90.xlsx'}")

    return df_check, df_mono


def build_threshold_with_interval(predictor, df_thresholds, df_monotonic=None):
    """
    为动态阈值构造 95% 置信区间

    Y_lower = Y_hat - 1.96 * sigma_Y, Y_upper = Y_hat + 1.96 * sigma_Y
    T_upper = T0 * exp(-Y_lower), T_mean = T0 * exp(-Y_hat), T_lower = T0 * exp(-Y_upper)

    Parameters
    ----------
    predictor : Predictor
    df_thresholds : pd.DataFrame (原始)
    df_monotonic : pd.DataFrame (可选，单调修正版)

    Returns
    -------
    df_interval : pd.DataFrame
    """
    print_section_header("构造动态阈值置信区间")

    sigma_Y = np.std(predictor.residuals)
    logger.info(f"残差标准差 sigma_Y = {sigma_Y:.6f}")

    rows = []

    for _, row in df_thresholds.iterrows():
        day = int(row['day'])
        for sample_type in SAMPLE_TYPES:
            col = f'{sample_type}_阈值'
            if col not in df_thresholds.columns:
                continue

            T_mean = row[col]
            T0 = get_static_threshold(sample_type)
            Y_hat = -np.log(max(T_mean, 1e-10) / T0)
            Y_lower = Y_hat - 1.96 * sigma_Y
            Y_upper = Y_hat + 1.96 * sigma_Y
            T_lower = T0 * np.exp(-Y_upper)
            T_upper = T0 * np.exp(-Y_lower)

            r = {
                'day': day,
                '样品类型': sample_type,
                'threshold_mean': T_mean,
                'threshold_lower_95': T_lower,
                'threshold_upper_95': T_upper,
            }

            if df_monotonic is not None and col in df_monotonic.columns:
                T_mono = df_monotonic[df_monotonic['day'] == day][col].values[0]
                Y_mono = -np.log(max(T_mono, 1e-10) / T0)
                Y_mono_lower = Y_mono - 1.96 * sigma_Y
                Y_mono_upper = Y_mono + 1.96 * sigma_Y
                r['threshold_mono_mean'] = T_mono
                r['threshold_mono_lower_95'] = T0 * np.exp(-Y_mono_upper)
                r['threshold_mono_upper_95'] = T0 * np.exp(-Y_mono_lower)

            rows.append(r)

    df_interval = pd.DataFrame(rows)

    output_dir = get_output_dir()
    df_interval.to_excel(output_dir / 'dynamic_threshold_with_interval_1_90.xlsx', index=False, engine='openpyxl')
    logger.info(f"阈值置信区间已保存: {output_dir / 'dynamic_threshold_with_interval_1_90.xlsx'}")

    return df_interval
