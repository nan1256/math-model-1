# -*- coding: utf-8 -*-
"""
数据预处理模块
处理缺失值、异常值、数据分组等
"""

import pandas as pd
import numpy as np
from .utils import setup_logger, print_section_header, print_dataframe_info, get_rust_grade, SAMPLE_TYPES

logger = setup_logger()

def preprocess_experiment_data(df):
    """
    预处理实验数据
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始实验数据，必须包含以下列：
        样品类型, 编号, 测量天数, 剩磁(mT), [温度(℃)], [相对湿度(%)]
    
    Returns:
    --------
    pd.DataFrame
        预处理后的数据
    """
    print_section_header("数据预处理")
    
    logger.info("开始实验数据预处理")
    
    # 1. 检查必要列
    required_cols = ['样品类型', '编号', '测量天数', '剩磁(mT)']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    
    # 2. 创建样品ID（组合样品类型和编号）
    df['样品ID'] = df['样品类型'] + '_' + df['编号'].astype(str)
    logger.info(f"创建样品ID: {df['样品ID'].nunique()}个样品")
    
    # 3. 检查缺失值
    logger.info("检查缺失值:")
    print_dataframe_info(df, "缺失值统计", logger)
    
    # 4. 移除缺失的关键列
    df = df.dropna(subset=['样品类型', '编号', '测量天数', '剩磁(mT)'])
    
    # 5. 数据类型转换
    df['编号'] = df['编号'].astype(int)
    df['测量天数'] = df['测量天数'].astype(int)
    df['剩磁(mT)'] = pd.to_numeric(df['剩磁(mT)'], errors='coerce')
    
    if '温度(℃)' in df.columns:
        df['温度(℃)'] = pd.to_numeric(df['温度(℃)'], errors='coerce')
    
    if '相对湿度(%)' in df.columns:
        df['相对湿度(%)'] = pd.to_numeric(df['相对湿度(%)'], errors='coerce')
    
    logger.info("数据类型转换完成")
    
    # 6. 检查异常值（剩磁值应该为正）
    invalid_mask = (df['剩磁(mT)'] <= 0)
    if invalid_mask.sum() > 0:
        logger.warning(f"发现{invalid_mask.sum()}个无效的剩磁值（≤0）")
        df = df[~invalid_mask]
    
    # 7. 提取初始剩磁 M(0)
    # 对每个样品，找到测量天数=0的记录作为初始值
    # 如果没有0天的记录，使用最早的记录作为初始值
    
    df = df.sort_values(['样品ID', '测量天数']).reset_index(drop=True)
    
    # 创建初始剩磁映射
    initial_remanence_dict = {}
    for sample_id in df['样品ID'].unique():
        sample_df = df[df['样品ID'] == sample_id]
        
        # 查找day=0的记录
        day_0 = sample_df[sample_df['测量天数'] == 0]
        if not day_0.empty:
            initial_remanence_dict[sample_id] = day_0['剩磁(mT)'].iloc[0]
        else:
            # 使用最早的记录
            initial_remanence_dict[sample_id] = sample_df.iloc[0]['剩磁(mT)']
    
    df['M0'] = df['样品ID'].map(initial_remanence_dict)
    
    logger.info(f"提取初始剩磁 M(0)，共{len(initial_remanence_dict)}个样品")
    logger.info(f"M(0) 范围: [{df['M0'].min():.4f}, {df['M0'].max():.4f}] mT")
    
    # 8. 添加锈蚀等级（根据样品类型）
    df['锈蚀等级'] = df['样品类型'].apply(get_rust_grade)
    
    logger.info("预处理完成")
    
    return df


def compute_remanence_ratio_and_decay(df):
    """
    计算剩磁保持率 R(t) 和累计衰减强度 Y(t)
    
    Parameters:
    -----------
    df : pd.DataFrame
        预处理后的数据
    
    Returns:
    --------
    pd.DataFrame
        添加了R和Y列的数据框
    """
    print_section_header("计算剩磁保持率和衰减强度")
    
    # 计算保持率 R(t) = M(t) / M(0)
    df['R'] = df['剩磁(mT)'] / df['M0']
    
    # 检查R的范围
    logger.info(f"保持率 R(t) 范围: [{df['R'].min():.4f}, {df['R'].max():.4f}]")
    
    # 标记无效的R值
    invalid_r_mask = (df['R'] <= 0)
    if invalid_r_mask.sum() > 0:
        logger.warning(f"发现{invalid_r_mask.sum()}个无效的R值（≤0），将被移除")
        df = df[~invalid_r_mask]
    
    # 计算衰减强度 Y(t) = -ln R(t)
    df['Y'] = -np.log(df['R'])
    
    logger.info(f"衰减强度 Y(t) 范围: [{df['Y'].min():.4f}, {df['Y'].max():.4f}]")
    logger.info(f"数据点数: {len(df)}")
    
    return df


def merge_weather_data(df_exp, df_weather):
    """
    将天气数据合并到实验数据
    
    Parameters:
    -----------
    df_exp : pd.DataFrame
        实验数据
    df_weather : pd.DataFrame
        天气数据
    
    Returns:
    --------
    pd.DataFrame
        合并后的数据
    """
    print_section_header("合并天气数据")
    
    if df_weather is None or df_weather.empty:
        logger.warning("天气数据为空，跳过合并，使用实验数据中的温湿度")
        df_exp = df_exp.copy()
        if '温度(℃)' in df_exp.columns:
            df_exp['温度'] = df_exp['温度(℃)']
        if '相对湿度(%)' in df_exp.columns:
            df_exp['相对湿度'] = df_exp['相对湿度(%)']
        return df_exp
    
    # 检查天气数据的列名
    if '天数' not in df_weather.columns:
        logger.warning("天气数据中缺少'天数'列")
        return df_exp
    
    # 合并
    df_merged = df_exp.merge(
        df_weather,
        left_on='测量天数',
        right_on='天数',
        how='left'
    )
    
    logger.info(f"合并完成，行数: {len(df_merged)}")
    
    # 对于没有对应天气数据的记录，需要从其他记录补充
    if df_merged.isnull().any().any():
        logger.warning("发现缺失的天气数据，尝试补充")
        
        if '温度' in df_merged.columns and df_merged['温度'].isnull().any():
            if '温度(℃)' in df_exp.columns:
                df_merged['温度'] = df_merged['温度'].fillna(df_exp['温度(℃)'])
        
        if '相对湿度' in df_merged.columns and df_merged['相对湿度'].isnull().any():
            if '相对湿度(%)' in df_exp.columns:
                df_merged['相对湿度'] = df_merged['相对湿度'].fillna(df_exp['相对湿度(%)'])
    
    return df_merged


def prepare_training_data(df):
    """
    准备训练数据
    
    移除 day=0 的数据（因为 Y = -ln R 在 day=0 时应严格为 0，
    包含 day=0 会人为引入截距偏差）；移除第23-29天之外缺失的测量点
    
    Parameters:
    -----------
    df : pd.DataFrame
        合并后的数据
    
    Returns:
    --------
    tuple : (训练数据, 预测天数列表)
    """
    print_section_header("准备训练数据")
    
    df_train = df.copy()
    
    n_before = len(df_train)
    df_train = df_train[df_train['测量天数'] > 0].copy()
    n_after = len(df_train)
    logger.info(f"移除 day=0 数据: {n_before} → {n_after} ({n_before - n_after} 条)")
    
    # 识别哪些天没有测量
    all_days = sorted(df_train['测量天数'].unique())
    logger.info(f"实测天数(day>0): {all_days}")
    
    logger.info(f"训练数据样本数: {len(df_train)}")
    logger.info(f"训练数据形状: {df_train.shape}")
    logger.info(f"样品类型分布:\n{df_train['样品类型'].value_counts()}")
    
    return df_train
