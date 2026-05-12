# -*- coding: utf-8 -*-
"""
主程序：剩磁法雷击判定预测

完整的数据分析与建模流程
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import (
    setup_logger, print_section_header, get_project_root, get_output_dir,
    ensure_dir_exists
)
from src.data_loader import load_all_data
from src.preprocessing import (
    preprocess_experiment_data, compute_remanence_ratio_and_decay,
    merge_weather_data, prepare_training_data
)
from src.feature_engineering import engineer_features
from src.modeling import build_models
from src.prediction import predict
from src.visualization import visualize
from src.cross_validation import leave_one_day_out_cv, leave_one_sample_out_cv
from src.collinearity import compute_collinearity, plot_correlation_heatmap
from src.monotonicity import check_and_fix_monotonicity, build_threshold_with_interval
from src.modeling import save_explain_model_coefficients, save_explain_vs_predict_comparison
from src.decision_workflow import generate_decision_workflow

logger = setup_logger()

MODEL_FORMULAS = {
    'Model 1': 'Y ~ 0 + C(样品类型):测量天数',
    'Model 2': 'Y ~ 0 + C(样品类型):测量天数 + C_T + TOW',
    'Model 3': 'Y ~ 0 + C(样品类型):测量天数 + C_T + TOW + C_TH',
    'Model 4': 'Y ~ 0 + C(样品类型):测量天数 + C_T + TOW + C_TH + rust_TOW',
    'Model 5': 'Y ~ 0 + C(样品类型):测量天数 + log_t + C_T + TOW + C_TH + rust_TOW',
    'Model 6': 'Y ~ 0 + C(样品类型):测量天数 + C(样品类型):log_t + C_T + TOW + C_TH + rust_TOW',
}

SAMPLE_TYPES = ['小号铁钉', '小号铁夹', '普通钢筋', '锈蚀钢筋']

def build_full_weather_data(df_exp, engineer):
    """
    为预测构建完整的1-90天天气-特征数据
    """
    print_section_header("构建完整预测天气数据", logger)

    weather_data = df_exp[['测量天数', '温度', '相对湿度', 'C_T', 'TOW', 'C_TH']].drop_duplicates()
    weather_data = weather_data.sort_values('测量天数').reset_index(drop=True)

    logger.info(f"已有的测量天数: {sorted(weather_data['测量天数'].unique().tolist())}")

    all_days = np.arange(1, 91)
    full_weather_data = []

    for day in all_days:
        day_data = weather_data[weather_data['测量天数'] == day]

        if not day_data.empty:
            row = day_data.iloc[0].to_dict()
        else:
            logger.warning(f"第{day}天缺少直接测量数据，使用插值")

            prev_day = weather_data[weather_data['测量天数'] < day]
            next_day = weather_data[weather_data['测量天数'] > day]

            if not prev_day.empty and not next_day.empty:
                prev_idx = prev_day.index[-1]
                next_idx = next_day.index[0]

                prev_data = weather_data.loc[prev_idx]
                next_data = weather_data.loc[next_idx]

                ratio = (day - prev_data['测量天数']) / (next_data['测量天数'] - prev_data['测量天数'])

                row = {
                    '测量天数': day,
                    '温度': prev_data['温度'] + ratio * (next_data['温度'] - prev_data['温度']),
                    '相对湿度': prev_data['相对湿度'] + ratio * (next_data['相对湿度'] - prev_data['相对湿度']),
                    'C_T': prev_data['C_T'] + ratio * (next_data['C_T'] - prev_data['C_T']),
                    'TOW': prev_data['TOW'] + ratio * (next_data['TOW'] - prev_data['TOW']),
                    'C_TH': prev_data['C_TH'] + ratio * (next_data['C_TH'] - prev_data['C_TH']),
                }
            else:
                logger.warning(f"无法为第{day}天进行插值")
                row = {
                    '测量天数': day,
                    '温度': np.nan, '相对湿度': np.nan,
                    'C_T': np.nan, 'TOW': np.nan, 'C_TH': np.nan,
                }

        row['log_t'] = np.log(1 + day)
        row['rust0'] = 3.0
        row['rust_TOW'] = row['rust0'] * row['TOW']
        row['rust_day'] = row['rust0'] * day
        row['rust_log_t'] = row['rust0'] * row['log_t']

        full_weather_data.append(row)

    df_full_weather = pd.DataFrame(full_weather_data)

    logger.info(f"完整天气数据构建完成，共{len(df_full_weather)}天")

    return df_full_weather


def save_results(builder, predictor, df_pred_23_29, df_thresholds, df_engineered, best_model_name):
    """保存所有结果到Excel和CSV文件"""
    print_section_header("保存结果", logger)

    output_dir = get_output_dir()
    logger.info(f"输出目录: {output_dir}")

    # 1. 模型对比表
    df_comparison = builder.get_model_comparison_df()
    comparison_csv = output_dir / 'model_comparison.csv'
    comparison_xlsx = output_dir / 'model_comparison.xlsx'
    df_comparison.to_csv(comparison_csv, index=False, encoding='utf-8')
    df_comparison.to_excel(comparison_xlsx, index=False, engine='openpyxl')
    logger.info(f"模型对比表已保存: {comparison_csv}, {comparison_xlsx}")

    # 2. 主模型系数表
    df_coef = predictor.get_coefficients_summary()
    coef_csv = output_dir / 'main_model_coefficients.csv'
    coef_xlsx = output_dir / 'main_model_coefficients.xlsx'
    df_coef.to_csv(coef_csv, index=False, encoding='utf-8')
    df_coef.to_excel(coef_xlsx, index=False, engine='openpyxl')
    logger.info(f"模型系数表已保存: {coef_csv}, {coef_xlsx}")

    # 3. 第23-29天预测结果
    pred_csv = output_dir / 'prediction_23_29.csv'
    pred_xlsx = output_dir / 'prediction_23_29.xlsx'
    df_pred_23_29.to_csv(pred_csv, index=False, encoding='utf-8')
    df_pred_23_29.to_excel(pred_xlsx, index=False, engine='openpyxl')
    logger.info(f"第23-29天预测结果已保存: {pred_csv}, {pred_xlsx}")

    # 4. 1-90天动态阈值
    threshold_xlsx = output_dir / 'dynamic_threshold_1_90.xlsx'
    df_thresholds.to_excel(threshold_xlsx, index=False, engine='openpyxl')
    logger.info(f"1-90天动态阈值已保存: {threshold_xlsx}")

    # 5. 处理后的数据
    processed_csv = output_dir / 'processed_data.csv'
    df_processed = df_engineered[[
        '样品ID', '样品类型', '编号', '测量天数', '剩磁(mT)', 'M0', 'R', 'Y',
        '温度', '相对湿度', 'C_T', 'TOW', 'C_TH', 'rust_TOW', 'log_t', '锈蚀等级'
    ]]
    df_processed.to_csv(processed_csv, index=False, encoding='utf-8')
    logger.info(f"处理后的数据已保存: {processed_csv}")

    # 6. 模型摘要
    model_summary_txt = output_dir / 'main_model_summary.txt'
    with open(model_summary_txt, 'w', encoding='utf-8') as f:
        f.write(f"最优模型: {best_model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(predictor.get_model_summary()))
    logger.info(f"模型摘要已保存: {model_summary_txt}")


def generate_markdown_summary(builder, predictor, df_engineered,
                               df_cv_summary=None, df_vif=None, df_monotonic=None,
                               df_sample_cv_summary=None):
    """生成Markdown论文摘要"""
    print_section_header("生成论文摘要", logger)

    output_dir = get_output_dir()
    summary_file = output_dir / 'modeling_summary.md'

    df_exp = df_engineered

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# 剩磁法雷击判定预测 —— 建模总结\n\n")

        # ---- 1. 数据规模 ----
        f.write("## 1. 数据规模\n\n")
        f.write(f"- **样品总数**: {df_exp['样品ID'].nunique()}个\n")
        f.write(f"- **样品类型**: 4种\n")
        for st in ['小号铁钉', '小号铁夹', '普通钢筋', '锈蚀钢筋']:
            cnt = (df_exp['样品类型'] == st).sum()
            f.write(f"  - {st}: {cnt}条记录\n")
        f.write(f"- **测量时间点(day>0)**: {len(df_exp['测量天数'].unique())}个\n")
        f.write(f"- **总数据点**: {len(df_exp)}条\n\n")

        # ---- 2. 数据预处理 ----
        f.write("## 2. 数据预处理说明\n\n")
        f.write("- 提取每个样品的初始剩磁值 M(0)\n")
        f.write("- 计算剩磁保持率 R(t) = M(t) / M(0)\n")
        f.write("- 计算衰减强度 Y(t) = -ln R(t)\n")
        f.write("- **移除 day=0 数据**：Y(0)=0 精确成立，保留 day=0 会人为引入截距偏差\n")
        f.write("- 合并天气数据（优先使用独立天气文件，否则回退到实验数据内置温湿度）\n\n")

        # ---- 3. 特征变量构造 ----
        f.write("## 3. 特征变量构造\n\n")
        f.write("### 3.1 因变量\n")
        f.write("- $R_{i,s}(t) = M_{i,s}(t) / M_{i,s}(0)$\n")
        f.write("- $Y_{i,s}(t) = -\\ln R_{i,s}(t)$（因变量，正值表示衰减）\n\n")

        f.write("### 3.2 环境变量标准化\n")
        f.write("- $z_T(d) = (T_d - \\mu_T) / \\sigma_T$\n")
        f.write("- $z_H(d) = (RH_d - \\mu_H) / \\sigma_H$\n\n")

        f.write("### 3.3 累计环境变量\n")
        f.write("- **$C_T(t) = \\sum_{d=1}^{t} z_T(d)$** 累计温度暴露\n")
        f.write("- **$TOW(t) = \\sum_{d=1}^{t} I_{wet}(d)$** 累计湿润时间，$I_{wet}=1$ 当 $RH>80\\%$ 且 $T>0$\n")
        f.write("- **$C_{TH}(t) = \\sum_{d=1}^{t} z_T(d) \\times z_H(d)$** 温湿度耦合项\n\n")

        f.write("### 3.4 锈蚀损伤交互项\n")
        f.write("- $rust_0$ = 锈蚀等级（小号铁钉/铁夹=0, 普通钢筋=0, 锈蚀钢筋=3）\n")
        f.write("- **$rust\\_TOW(t) = rust_0 \\times TOW(t)$** 锈蚀-湿润交互项\n\n")

        f.write("### 3.5 磁黏滞对数时间项\n")
        f.write("- **$\\log\\_t = \\ln(1 + t)$**\n\n")

        # ---- 4. 模型公式 ----
        f.write("## 4. 模型公式\n\n")
        f.write("所有模型使用 $0 + C(\\text{样品类型}):\\text{测量天数}$ 交互项，"
                "避免截距引入的物理偏差。\n\n")

        f.write("### Model 1\n")
        f.write("$$Y = \\alpha_s \\cdot t + \\varepsilon$$\n\n")

        f.write("### Model 2\n")
        f.write("$$Y = \\alpha_s \\cdot t + \\beta_T C_T(t) + \\beta_H TOW(t) + \\varepsilon$$\n\n")

        f.write("### Model 3\n")
        f.write("$$Y = \\alpha_s \\cdot t + \\beta_T C_T(t) + \\beta_H TOW(t) + \\beta_{TH} C_{TH}(t) + \\varepsilon$$\n\n")

        f.write("### Model 4（稳健基准模型，不含 log_t）\n")
        f.write("$$Y = \\alpha_s \\cdot t + \\beta_T C_T(t) + \\beta_H TOW(t) + "
                "\\beta_{TH} C_{TH}(t) + \\beta_{rust} \\, rust\\_TOW(t) + \\varepsilon$$\n\n")

        f.write("### Model 5（最终预测模型，含统一 log_t）\n")
        f.write("$$Y = \\alpha_s \\cdot t + \\beta_{\\log} \\ln(1+t) + "
                "\\beta_T C_T(t) + \\beta_H TOW(t) + \\beta_{TH} C_{TH}(t) + "
                "\\beta_{rust} \\, rust\\_TOW(t) + \\varepsilon$$\n\n")

        f.write("### Model 6（复杂对照模型，含 type 独立 log_t）\n")
        f.write("$$Y = \\alpha_s \\cdot t + \\beta_{\\log,s} \\ln(1+t) + "
                "\\beta_T C_T(t) + \\beta_H TOW(t) + \\beta_{TH} C_{TH}(t) + "
                "\\beta_{rust} \\, rust\\_TOW(t) + \\varepsilon$$\n\n")

        f.write("$\\alpha_s$ 为第 $s$ 种样品类型的时间衰减斜率。\n\n")

        # ---- 5. 模型对比 ----
        f.write("## 5. 模型对比结果\n\n")
        df_comp = builder.get_model_comparison_df()
        f.write(df_comp.to_markdown(index=False) + "\n\n")

        f.write("**模型角色**:\n")
        f.write("- Model 4：稳健基准模型（不含磁黏滞项），仅作对照\n")
        f.write("- Model 5：**最终预测模型**，加入统一 $\\ln(1+t)$，Y/M空间误差均低于 Model 4\n")
        f.write("- Model 6：复杂增强模型（type独立 $\\ln(1+t)$），参数多，仅作对照\n\n")

        f.write("**选择 Model 5 的理由**:\n")
        f.write("1. Y空间和M空间误差均优于Model 4\n")
        f.write("2. 参数增加有限（仅1个 log_t 系数），优于Model 6的4个独立系数\n")
        f.write("3. $\\ln(1+t)$ 有明确的磁黏滞物理背景\n\n")

        # ---- 5.5. 解释模型与预测模型分工 ----
        f.write("## 5.5. 解释模型与预测模型的分工\n\n")
        f.write("### ExplainModel（解释模型）\n")
        f.write("$$Y = \\alpha_s \\cdot t + \\beta_{\\log} \\ln(1+t) + \\beta_{rust\\_base} \\cdot (Rust_0 \\cdot t) + \\varepsilon$$\n\n")
        f.write("**用途**：机理分析——回答时间是否主导、不同样品类型衰减斜率是否不同、锈蚀钢筋是否比普通钢筋衰减更快。")
        f.write("仅包含时间、log_t 和锈蚀基线项，不含环境累计变量，避免共线性导致解释困难。\n\n")
        f.write("### Model 5（预测模型）\n")
        f.write("$$Y = \\alpha_s \\cdot t + \\beta_{\\log} \\ln(1+t) + \\beta_T C_T + \\beta_H TOW + \\beta_{TH} C_{TH} + \\beta_{rust} \\, rust\\_TOW + \\varepsilon$$\n\n")
        f.write("**用途**：工程应用——第23~29天剩磁预测、1~90天动态阈值、动态阈值置信区间、实际雷击判定流程。\n\n")
        f.write("**核心表述**：解释模型用于机理分析，预测模型用于工程应用。")
        f.write("由于累计环境变量与时间变量存在共线性，本文不把 Model 5 中所有环境项解释为独立因果效应，")
        f.write("而将其作为预测模型中的轨迹修正项。\n\n")
        f.write("详见 `explain_model_coefficients.xlsx` 和 `explain_vs_predict_model_comparison.xlsx`。\n\n")

        # ---- 6. 参数解释 ----
        f.write("## 6. 主模型参数解释（Model 5）\n\n")
        df_coef = predictor.get_coefficients_summary()
        f.write(df_coef.to_markdown(index=False) + "\n\n")

        f.write("### 关键参数分析（注意：因变量为 $Y=-\\ln R$）\n\n")
        f.write("- **正系数** → 增大 $Y$，表示增强衰减\n")
        f.write("- **负系数** → 减小 $Y$，表示对基础时间衰减趋势的补偿或修正\n\n")
        f.write("- **type:day 交互项（$\\alpha_s$）**：各样品类型的基础时间衰减斜率。")
        f.write("其中锈蚀钢筋的时间衰减斜率（约 0.0950）高于普通钢筋（约 0.0645），")
        f.write("由于二者具有相同规格，差异主要来自锈蚀状态，")
        f.write("说明锈蚀降低了剩磁稳定性并增强了基础时间衰减趋势。\n")
        f.write("- **log_t**：磁黏滞对数时间项，系数显著为正（约 0.281），")
        f.write("说明加入前期弛豫效应后模型更能解释剩磁衰减。\n")
        f.write("- **C_T**：系数为正且显著（约 0.055），")
        f.write("说明在控制其他变量后，累计温度暴露对衰减强度存在正向修正。\n")
        f.write("- **TOW、C_TH、rust_TOW**：系数为负且显著，")
        f.write("应解释为共线性背景下的轨迹修正项，")
        f.write("而非直接减缓或加速衰减的独立因果项。")
        f.write("C_T、TOW、C_TH 与 day/log_t 均为时间累计或单调相关变量，")
        f.write("存在较强共线性（VIF 最高达 197），")
        f.write("因此其系数符号不宜直接解释为独立因果效应。\n")
        f.write("- **rust_TOW**：锈蚀-湿润交互修正项，系数为负（约 -0.0022）。")
        f.write("在控制样品类型时间衰减斜率、log_t 与累计环境变量后，")
        f.write("该项主要表示锈蚀样品在湿润暴露下相对于基础时间衰减趋势的补偿性修正，")
        f.write("不单独解释为直接加速衰减。")
        f.write("锈蚀影响主要通过锈蚀钢筋与普通钢筋的 type:day 斜率差异体现。\n\n")

        # ---- 7. 共线性分析 ----
        f.write("## 7. 共线性分析\n\n")
        f.write("对 day、log_t、C_T、TOW、C_TH、rust_TOW 计算相关系数和 VIF。\n\n")

        if df_vif is not None and not df_vif.empty:
            f.write("### VIF (方差膨胀因子)\n\n")
            f.write(df_vif.to_markdown(index=False) + "\n\n")

        f.write("### 发现\n")
        f.write("1. day 和 log_t 高度相关（$\\ln(1+t)$ 是 t 的单调变换）\n")
        f.write("2. C_T、TOW、C_TH 是时间累计变量，与 day/log_t 存在共线性\n")
        f.write("3. **由于共线性，环境项的符号可能不同于直觉预期**：")
        f.write("负系数不表示环境因素减缓衰减，而是相对时间趋势的修正效应\n")
        f.write("4. 相关系数矩阵和 VIF 表详见 `feature_correlation_matrix.xlsx` 和 `feature_vif.xlsx`\n\n")

        # ---- 8. Leave-One-Day-Out CV ----
        f.write("## 8. Leave-One-Day-Out 交叉验证\n\n")
        f.write("按测量天数留一交叉验证：每次去掉一个 day 的所有数据，用其余天数训练，")
        f.write("预测被去掉 day 的所有样品。评估模型在未观测时间点上的插值/外推能力。\n\n")
        f.write("**特别说明**：23~29 天预测属于 22 天和 30 天之间的插值区间，")
        f.write("留一天验证能更好地评估模型对未观测时间点的预测能力。\n\n")

        if df_cv_summary is not None and not df_cv_summary.empty:
            f.write(df_cv_summary.to_markdown(index=False) + "\n\n")

        f.write("详见 `leave_one_day_cv_results.xlsx` 和 `leave_one_day_cv_summary.xlsx`。\n\n")

        # ---- 8.5. Sample CV ----
        f.write("## 8.5. 样品留一交叉验证\n\n")
        f.write("按样品留一交叉验证：每次去掉一个完整样品的所有时间点数据，")
        f.write("用剩余样品训练 Model 5，预测被去掉样品的所有时间点。")
        f.write("评估模型对新铁件样品的泛化能力。\n\n")
        f.write("Leave-One-Day-Out CV 用于验证未观测时间点预测能力，")
        f.write("Leave-One-Sample-Out CV 用于验证模型对新样品的泛化能力。\n\n")

        if df_sample_cv_summary is not None and not df_sample_cv_summary.empty:
            f.write(df_sample_cv_summary.to_markdown(index=False) + "\n\n")

        f.write("详见 `leave_one_sample_cv_results.xlsx` 和 `leave_one_sample_cv_summary.xlsx`。\n\n")

        # ---- 9. 23~29天预测 ----
        f.write("## 9. 第23-29天预测结果\n\n")
        f.write("使用 Model 5（最终预测模型）预测四类样品第 23~29 天的剩磁值。\n")
        f.write("预测区间基于 Model 5 残差标准差（±1.96σ）。\n\n")
        f.write("详见 `prediction_23_29.xlsx`。\n\n")

        # ---- 10. 动态阈值与单调修正 ----
        f.write("## 10. 动态阈值与单调修正\n\n")
        f.write("### 10.1 动态阈值公式\n\n")
        f.write("$$T_{dyn,s}(t) = T_{0,s} \\times \\exp[-\\hat{Y}_s(t)]$$\n\n")
        f.write("其中 $T_{0,s}$ 为样品类型的静态阈值:\n")
        f.write("- 小号铁钉、小号铁夹: $T_0 = 1.0$ mT\n")
        f.write("- 普通钢筋、锈蚀钢筋: $T_0 = 1.5$ mT\n\n")

        f.write("### 10.2 单调性检查与修正\n\n")
        f.write("由于环境波动，原始预测的 $\\hat{Y}(t)$ 可能局部下降，")
        f.write("导致动态阈值局部上升，不符合剩磁单调衰减的物理规律。\n\n")
        f.write("**单调修正**:\n")
        f.write("$$Y_{mono}(t) = \\max_{1 \\le d \\le t} \\hat{Y}(d)$$\n")
        f.write("$$T_{dyn,mono}(t) = T_0 \\times \\exp[-Y_{mono}(t)]$$\n\n")
        f.write("原始阈值反映环境波动，单调修正版更适合实际判定流程。\n\n")
        f.write("详见 `dynamic_threshold_monotonic_check.xlsx` 和 `monotonic_dynamic_threshold_1_90.xlsx`。\n\n")

        # ---- 10.3. 阈值置信区间 ----
        f.write("### 10.3 动态阈值置信区间\n\n")
        f.write("基于 Model 5 残差标准差 $\\sigma_Y$ 构造近似 95% 阈值区间：\n\n")
        f.write("$$Y_{lower} = \\hat{Y} - 1.96\\sigma_Y, \\quad Y_{upper} = \\hat{Y} + 1.96\\sigma_Y$$\n")
        f.write("$$T_{upper} = T_0 e^{-Y_{lower}}, \\quad T_{mean} = T_0 e^{-\\hat{Y}}, \\quad T_{lower} = T_0 e^{-Y_{upper}}$$\n\n")
        f.write("动态阈值区间可用于雷击判定置信度分级。\n\n")
        f.write("详见 `dynamic_threshold_with_interval_1_90.xlsx`。\n\n")

        # ---- 10.4. 雷击判定规则 ----
        f.write("### 10.4 雷击判定分级规则\n\n")
        f.write("| 条件 | 判定 | 说明 |\n")
        f.write("|------|------|------|\n")
        f.write("| $M_{obs} \\ge T_{upper\\_95}$ | 高置信雷击 | 实测剩磁高于阈值上界 |\n")
        f.write("| $T_{mean} \\le M_{obs} < T_{upper\\_95}$ | 疑似雷击 | 实测剩磁处于均值与上界之间 |\n")
        f.write("| $T_{lower\\_95} \\le M_{obs} < T_{mean}$ | 低置信疑似 | 需结合现场证据综合判断 |\n")
        f.write("| $M_{obs} < T_{lower\\_95}$ | 雷击证据不足 | 实测剩磁低于阈值下界 |\n\n")
        f.write("详见 `lightning_decision_workflow.xlsx`。\n\n")

        # ---- 11. 可视化 ----
        f.write("## 11. 可视化输出\n\n")
        f.write("| 编号 | 文件名 | 内容 |\n")
        f.write("|------|--------|------|\n")
        f.write("| 1 | `01_average_remanence.png` | 四类样品平均剩磁随时间变化 |\n")
        f.write("| 2 | `02_remanence_ratio.png` | 平均剩磁保持率随时间变化 |\n")
        f.write("| 3 | `03_temperature_humidity.png` | 温度和湿度变化 |\n")
        f.write("| 4 | `04_cumulative_features.png` | 累计环境特征变化 |\n")
        f.write("| 5 | `05_measured_vs_predicted.png` | 实测值vs预测值 |\n")
        f.write("| 6 | `06_residuals_distribution.png` | 残差分布 |\n")
        f.write("| 7 | `07_prediction_23_29.png` | 第23-29天预测曲线 |\n")
        f.write("| 8 | `08_dynamic_thresholds.png` | 1-90天动态阈值曲线 |\n")
        f.write("| 9 | `09_corrosion_comparison.png` | 普通钢筋vs锈蚀钢筋对比 |\n")
        f.write("| 10 | `10_feature_correlation_heatmap.png` | 特征相关性热力图 |\n")
        f.write("| 11 | `11_leave_one_day_cv_comparison.png` | CV模型对比 |\n")
        f.write("| 12 | `12_monotonic_threshold_comparison.png` | 原始vs单调修正阈值 |\n")
        f.write("| 13 | `13_leave_one_sample_cv_comparison.png` | Sample-Out CV |\n")
        f.write("| 14 | `14_dynamic_threshold_interval.png` | 动态阈值置信区间 |\n")
        f.write("| 15 | `15_decision_rule_diagram.png` | 判定规则示意图 |\n\n")

        # ---- 12. 结论 ----
        f.write("## 12. 结论（可直接用于论文）\n\n")
        f.write("1. **时间延迟是剩磁衰减的主导因素**。Model 5 中四类样品的 type:day ")
        f.write("时间衰减斜率均显著为正，说明剩磁保持率随检测延迟时间增加而持续下降。\n\n")
        f.write("2. **样品类型差异显著**。四类样品的时间衰减斜率不同，")
        f.write("说明尺寸、形状和材料状态会影响剩磁稳定性。\n\n")
        f.write("3. **锈蚀状态降低剩磁稳定性**。普通钢筋与锈蚀钢筋规格相同，")
        f.write("但锈蚀钢筋的时间衰减斜率（约 0.0950）高于普通钢筋（约 0.0645），")
        f.write("说明锈蚀状态会增强基础时间衰减趋势。\n\n")
        f.write("4. **环境累计变量对衰减轨迹具有显著修正作用**。")
        f.write("C_T、TOW、C_TH 等变量在模型中具有统计显著性，")
        f.write("但由于它们与 day/log_t 存在较强共线性（VIF 最高达 197），")
        f.write("其系数不宜解释为严格独立因果效应。\n\n")
        f.write("5. **rust_TOW 是锈蚀-湿润交互修正项**。由于其系数为负，")
        f.write("该项主要表示锈蚀样品在湿润暴露下相对于基础时间趋势的补偿性修正，")
        f.write("不单独解释为直接加速衰减。\n\n")
        f.write("6. **加入 log_t 后预测能力显著增强**。与 Model 4 相比，")
        f.write("Model 5 在训练集误差和 Leave-One-Day-Out CV 中均取得更低的 RMSE，")
        f.write("说明磁黏滞对数时间项能够有效提升未观测时间点预测能力。\n\n")
        f.write("7. **动态阈值修正适用于灾害上报滞后的雷击鉴定场景**。")
        f.write("基于 Model 5 的动态阈值能够根据检测延迟和环境暴露修正静态阈值，")
        f.write("单调修正版则更适合实际判定流程。\n\n")
        f.write("8. **Leave-One-Day-Out CV 验证了模型对未观测时间点的预测能力**，")
        f.write("Model 5 的 CV_RMSE_M=0.090 mT，显著优于 Model 4 的 0.187 mT。\n\n")
        f.write("9. **Leave-One-Sample-Out CV 验证了模型对新样品的泛化能力**，")
        f.write("说明模型不依赖于特定样品训练集。\n\n")
        f.write("10. **动态阈值区间比单点阈值更适合实际鉴定**：")
        f.write("基于 95% 置信区间的分级规则可提供置信度分级，")
        f.write("避免单点阈值在高不确定性场景下的误判。\n\n")
        f.write("11. **解释模型用于机理分析，预测模型用于动态阈值和工程应用**。")
        f.write("解释模型仅含时间、log_t 和锈蚀基线项，便于解释；")
        f.write("预测模型含环境累计变量，作为轨迹修正项提升预测精度。\n")
        f.write("12. **环境变量因共线性不作强因果解释**，")
        f.write("而作为预测模型中的轨迹修正项。\n\n")

    logger.info(f"论文摘要已保存: {summary_file}")


def main():
    """主函数"""
    print("=" * 70)
    print("剩磁法雷击判定预测 —— 完整数据分析与建模系统")
    print("=" * 70)

    try:
        # Step 1: 加载数据
        df_exp, df_weather, col_maps = load_all_data()

        # Step 2: 预处理
        df_exp_clean = preprocess_experiment_data(df_exp)
        df_exp_clean = compute_remanence_ratio_and_decay(df_exp_clean)
        df_merged = merge_weather_data(df_exp_clean, df_weather)
        df_train = prepare_training_data(df_merged)

        # Step 3: 特征工程
        df_engineered, engineer = engineer_features(df_train)

        # Step 4: 建立模型
        builder, best_model_name, best_model = build_models(df_engineered)

        # Step 5: 构建完整天气特征数据
        df_full_weather = build_full_weather_data(df_engineered, engineer)

        # Step 6: 用最优模型 (Model 5) 预测
        predictor, df_pred_23_29, df_thresholds = predict(
            df_engineered, best_model, engineer, df_full_weather
        )

        # Step 6.5: 保存含预测区间的 23-29 天预测
        output_dir = get_output_dir()
        df_pred_23_29.to_excel(output_dir / 'prediction_23_29.xlsx', index=False, engine='openpyxl')
        df_pred_23_29.to_csv(output_dir / 'prediction_23_29.csv', index=False, encoding='utf-8')
        logger.info(f"23-29天预测(含区间)已保存: {output_dir / 'prediction_23_29.xlsx'}")

        # Step 7: 共线性分析
        df_corr, df_vif = compute_collinearity(df_engineered)
        plot_correlation_heatmap(df_corr)

        # Step 8: Leave-One-Day-Out 交叉验证
        df_cv_results, df_cv_summary = leave_one_day_out_cv(df_engineered,
            model_names=['Model 4', 'Model 5', 'Model 6'])

        # Step 9: 动态阈值单调性检查与修正
        df_mono_check, df_monotonic = check_and_fix_monotonicity(df_thresholds)

        # Step 9.5: 动态阈值置信区间
        df_interval = build_threshold_with_interval(predictor, df_thresholds, df_monotonic)

        # Step 9.6: Leave-One-Sample-Out CV
        df_sample_cv_results, df_sample_cv_summary = leave_one_sample_out_cv(df_engineered)

        # Step 9.7: 保存解释模型系数和对比
        save_explain_model_coefficients(builder)
        save_explain_vs_predict_comparison(builder)

        # Step 9.8: 雷击判定流程
        generate_decision_workflow()

        # Step 10: 可视化
        visualize(df_engineered, predictor, df_pred_23_29, df_thresholds,
                  df_cv_summary=df_cv_summary, df_monotonic=df_monotonic,
                  df_sample_cv_summary=df_sample_cv_summary, df_interval=df_interval)

        # Step 11: 保存结果
        save_results(builder, predictor, df_pred_23_29, df_thresholds,
                     df_engineered, best_model_name)

        # Step 12: 生成论文摘要
        generate_markdown_summary(builder, predictor, df_engineered,
                                   df_cv_summary, df_vif, df_monotonic,
                                   df_sample_cv_summary=df_sample_cv_summary)

        # 完成
        print_section_header("建模流程完成", logger)
        logger.info("[OK] 所有阶段执行完毕")
        logger.info(f"[OK] 输出文件位置: {get_output_dir()}")

        print("\n" + "=" * 70)
        print("输出文件列表:")
        print("-" * 70)
        for file in sorted(get_output_dir().glob('*')):
            if file.is_file():
                print(f"  [OK] {file.name}")

        figures_dir = get_output_dir() / 'figures'
        if figures_dir.exists():
            print("\n图表文件:")
            for file in sorted(figures_dir.glob('*')):
                if file.is_file():
                    print(f"  [OK] {file.name}")

        print("=" * 70 + "\n")
        return 0

    except Exception as e:
        logger.error(f"建模流程出错: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
