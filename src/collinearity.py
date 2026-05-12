# -*- coding: utf-8 -*-
"""
共线性分析模块

计算特征间的相关系数矩阵和方差膨胀因子 (VIF)
并输出相关性热力图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .utils import setup_logger, print_section_header, get_output_dir

mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

logger = setup_logger()

FEATURE_LABELS = {
    'day': '测量天数 t',
    'log_t': 'ln(1+t)',
    'C_T': 'C_T',
    'TOW': 'TOW',
    'C_TH': 'C_TH',
    'rust_TOW': 'rust_TOW',
}


def compute_collinearity(df):
    """
    计算特征相关系数矩阵和 VIF

    Parameters
    ----------
    df : pd.DataFrame
        特征工程后的数据，需包含 day, log_t, C_T, TOW, C_TH, rust_TOW

    Returns
    -------
    df_corr : pd.DataFrame
        相关系数矩阵
    df_vif : pd.DataFrame
        VIF 表
    """
    print_section_header("共线性分析")

    features = ['day', 'log_t', 'C_T', 'TOW', 'C_TH', 'rust_TOW']
    feature_cols = {'day': '测量天数', 'log_t': 'log_t', 'C_T': 'C_T',
                    'TOW': 'TOW', 'C_TH': 'C_TH', 'rust_TOW': 'rust_TOW'}
    available = [f for f in features if f in df.columns or feature_cols.get(f) in df.columns]

    if '测量天数' in df.columns and 'day' not in df.columns:
        df['day'] = df['测量天数']

    data = df[[c for c in ['day', 'log_t', 'C_T', 'TOW', 'C_TH', 'rust_TOW'] if c in df.columns]].dropna()
    if len(data) == 0:
        logger.warning("无有效特征数据")
        return pd.DataFrame(), pd.DataFrame()

    # 相关系数矩阵
    df_corr = data.corr()
    logger.info("相关系数矩阵:")
    logger.info(f"\n{df_corr.to_string()}")

    # VIF
    vif_data = []
    col_names = data.columns.tolist()
    for i, col in enumerate(col_names):
        try:
            vif = variance_inflation_factor(data.values, i)
        except Exception:
            vif = np.nan
        vif_data.append({'Feature': FEATURE_LABELS.get(col, col), 'VIF': vif})
    df_vif = pd.DataFrame(vif_data)
    logger.info("VIF:")
    logger.info(f"\n{df_vif.to_string()}")

    output_dir = get_output_dir()
    df_corr.to_excel(output_dir / 'feature_correlation_matrix.xlsx', engine='openpyxl')
    df_vif.to_excel(output_dir / 'feature_vif.xlsx', index=False, engine='openpyxl')
    logger.info(f"共线性结果已保存到 {output_dir}")

    return df_corr, df_vif


def plot_correlation_heatmap(df_corr):
    """绘制特征相关性热力图（图10）"""
    print_section_header("绘制相关性热力图")

    if df_corr.empty:
        logger.warning("相关性矩阵为空，跳过绘图")
        return

    output_dir = get_output_dir() / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(df_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)

    labels = [FEATURE_LABELS.get(c, c) for c in df_corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = df_corr.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if abs(val) > 0.5 else 'black')

    ax.set_title('特征相关性热力图', fontsize=14, fontweight='bold')
    fig.tight_layout()

    output_path = output_dir / '10_feature_correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"已保存: {output_path}")
    plt.close()
