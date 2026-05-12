# -*- coding: utf-8 -*-
"""
雷击判定流程模块
"""

import pandas as pd
from .utils import setup_logger, print_section_header, get_output_dir

logger = setup_logger()


def generate_decision_workflow():
    """生成雷击判定流程说明表"""
    print_section_header("生成雷击判定流程")

    steps = [
        {'step': 1, 'input': '样品类型、样品编号、检测延迟天数 t、实测剩磁 M_obs、温湿度数据',
         'method': '数据预处理',
         'output': '标准化温湿度、累计环境变量 C_T, TOW, C_TH',
         'description': '根据检测延迟天数范围计算累计温度暴露、湿润时间和温湿度耦合项'},

        {'step': 2, 'input': '样品类型、检测延迟天数 t、C_T, TOW, C_TH',
         'method': '调用 Model5 预测 Y_hat',
         'output': '预测衰减强度 Y_hat(t)',
         'description': 'Y = alpha_s*t + beta_log*ln(1+t) + beta_T*C_T + beta_H*TOW + beta_TH*C_TH + beta_rust*rust_TOW'},

        {'step': 3, 'input': 'Y_hat(t)、样品类型 T0',
         'method': '计算动态阈值及 95% 区间',
         'output': 'threshold_mean, threshold_lower_95, threshold_upper_95',
         'description': 'T_dyn = T0*exp(-Y_hat)；区间基于残差标准差 sigma_Y 构造'},

        {'step': 4, 'input': 'M_obs、threshold_mean、threshold_lower_95、threshold_upper_95',
         'method': '分级比较',
         'output': '判定等级',
         'description': '见步骤5的分级规则'},

        {'step': 5, 'input': '判定等级',
         'method': '判定结论',
         'output': '雷击置信说明',
         'description': (
             '规则1: M_obs >= threshold_upper_95 → 高置信雷击；'
             '规则2: threshold_mean <= M_obs < threshold_upper_95 → 疑似雷击；'
             '规则3: threshold_lower_95 <= M_obs < threshold_mean → 低置信疑似，需结合现场证据；'
             '规则4: M_obs < threshold_lower_95 → 雷击证据不足'
         )},
    ]

    df = pd.DataFrame(steps)

    output_dir = get_output_dir()
    df.to_excel(output_dir / 'lightning_decision_workflow.xlsx', index=False, engine='openpyxl')
    logger.info(f"判定流程已保存: {output_dir / 'lightning_decision_workflow.xlsx'}")

    return df
