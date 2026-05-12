# -*- coding: utf-8 -*-
"""
交叉验证模块 —— Leave-One-Day-Out CV

每次去掉一个测量天数的所有样品，用其余天数训练，预测被去掉的天数。
评估模型在未观测时间点上的插值/外推能力。
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.formula.api import ols
from .utils import setup_logger, print_section_header, get_output_dir

logger = setup_logger()

MODEL_CONFIGS = {
    'Model 4': 'Y ~ 0 + C(样品类型):测量天数 + C_T + TOW + C_TH + rust_TOW',
    'Model 5': 'Y ~ 0 + C(样品类型):测量天数 + log_t + C_T + TOW + C_TH + rust_TOW',
    'Model 6': 'Y ~ 0 + C(样品类型):测量天数 + C(样品类型):log_t + C_T + TOW + C_TH + rust_TOW',
}


def leave_one_day_out_cv(df, model_names=None):
    """
    按测量天数留一交叉验证

    对每个模型，依次去掉一个 day 的所有数据，
    用剩余数据训练，预测被去掉 day 的所有样品。

    Parameters
    ----------
    df : pd.DataFrame
        特征工程后的训练数据
    model_names : list, optional
        要评估的模型名称列表

    Returns
    -------
    df_results : pd.DataFrame
        每天、每模型的详细预测结果
    df_summary : pd.DataFrame
        每模型的汇总指标
    """
    print_section_header("Leave-One-Day-Out 交叉验证")

    if model_names is None:
        model_names = ['Model 4', 'Model 5', 'Model 6']

    unique_days = sorted(df['测量天数'].unique())
    logger.info(f"共{len(unique_days)}个不同测量天数: {unique_days}")

    all_results = []

    for model_name in model_names:
        formula = MODEL_CONFIGS.get(model_name)
        if formula is None:
            logger.warning(f"未知模型: {model_name}，跳过")
            continue

        logger.info(f"--- {model_name} ---")

        for held_out_day in unique_days:
            train_df = df[df['测量天数'] != held_out_day].copy()
            test_df = df[df['测量天数'] == held_out_day].copy()

            if len(test_df) == 0:
                continue

            try:
                model = ols(formula, data=train_df).fit()
            except Exception as e:
                logger.warning(f"  {model_name} day={held_out_day}: 训练失败 ({e})")
                continue

            y_pred_Y = model.predict(test_df).values
            y_true_Y = test_df['Y'].values

            M0 = test_df['M0'].values
            M_true = M0 * np.exp(-y_true_Y)
            M_pred = M0 * np.exp(-y_pred_Y)

            for i, (_, row) in enumerate(test_df.iterrows()):
                all_results.append({
                    'Model': model_name,
                    'HeldOut_Day': held_out_day,
                    '样品ID': row.get('样品ID', ''),
                    '样品类型': row.get('样品类型', ''),
                    '测量天数': row['测量天数'],
                    'Y_true': y_true_Y[i],
                    'Y_pred': y_pred_Y[i],
                    'M_true': M_true[i],
                    'M_pred': M_pred[i],
                })

    df_results = pd.DataFrame(all_results)

    summary_rows = []
    for model_name in model_names:
        model_df = df_results[df_results['Model'] == model_name]
        if len(model_df) == 0:
            continue

        rmse_Y = np.sqrt(mean_squared_error(model_df['Y_true'], model_df['Y_pred']))
        mae_Y = mean_absolute_error(model_df['Y_true'], model_df['Y_pred'])
        rmse_M = np.sqrt(mean_squared_error(model_df['M_true'], model_df['M_pred']))
        mae_M = mean_absolute_error(model_df['M_true'], model_df['M_pred'])
        n_days = model_df['HeldOut_Day'].nunique()

        summary_rows.append({
            'Model': model_name,
            'CV_RMSE_Y': rmse_Y,
            'CV_MAE_Y': mae_Y,
            'CV_RMSE_M': rmse_M,
            'CV_MAE_M': mae_M,
            'N_HeldOut_Days': n_days,
            'N_Predictions': len(model_df),
        })

        logger.info(f"{model_name}: CV_RMSE_Y={rmse_Y:.4f}, CV_MAE_Y={mae_Y:.4f}, "
                    f"CV_RMSE_M={rmse_M:.4f}, CV_MAE_M={mae_M:.4f}")

    df_summary = pd.DataFrame(summary_rows)

    output_dir = get_output_dir()
    df_results.to_csv(output_dir / 'leave_one_day_cv_results.csv', index=False, encoding='utf-8')
    df_results.to_excel(output_dir / 'leave_one_day_cv_results.xlsx', index=False, engine='openpyxl')
    df_summary.to_csv(output_dir / 'leave_one_day_cv_summary.csv', index=False, encoding='utf-8')
    df_summary.to_excel(output_dir / 'leave_one_day_cv_summary.xlsx', index=False, engine='openpyxl')

    logger.info(f"CV结果已保存到 {output_dir}")

    return df_results, df_summary


def leave_one_sample_out_cv(df, model_name='Model 5'):
    """
    按样品留一交叉验证（Leave-One-Sample-Out CV）

    每次去掉一个 sample_key（样品类型+编号）的全部时间点数据，
    用剩余样品训练，预测被去掉样品的所有时间点。
    评估模型对新铁件样品的泛化能力。

    Parameters
    ----------
    df : pd.DataFrame
        特征工程后的训练数据
    model_name : str
        模型名称

    Returns
    -------
    df_results : pd.DataFrame
        每样品的详细预测结果
    df_summary : pd.DataFrame
        汇总指标
    """
    print_section_header("Leave-One-Sample-Out 交叉验证")

    formula = MODEL_CONFIGS.get(model_name)
    if formula is None:
        logger.error(f"未知模型: {model_name}")
        return pd.DataFrame(), pd.DataFrame()

    df['sample_key'] = df['样品类型'].astype(str) + '_' + df['编号'].astype(str)
    unique_samples = sorted(df['sample_key'].unique())
    logger.info(f"共{len(unique_samples)}个独立样品")

    all_results = []

    for held_out_sample in unique_samples:
        train_df = df[df['sample_key'] != held_out_sample].copy()
        test_df = df[df['sample_key'] == held_out_sample].copy()

        if len(test_df) == 0:
            continue

        try:
            model = ols(formula, data=train_df).fit()
        except Exception as e:
            logger.warning(f"  {model_name} sample={held_out_sample}: 训练失败 ({e})")
            continue

        y_pred_Y = model.predict(test_df).values
        y_true_Y = test_df['Y'].values
        M0 = test_df['M0'].values
        M_true = M0 * np.exp(-y_true_Y)
        M_pred = M0 * np.exp(-y_pred_Y)

        for i, (_, row) in enumerate(test_df.iterrows()):
            all_results.append({
                'Model': model_name,
                'HeldOut_Sample': held_out_sample,
                '样品类型': row.get('样品类型', ''),
                '测量天数': row['测量天数'],
                'Y_true': y_true_Y[i],
                'Y_pred': y_pred_Y[i],
                'M_true': M_true[i],
                'M_pred': M_pred[i],
            })

    df_results = pd.DataFrame(all_results)

    model_df = df_results[df_results['Model'] == model_name]
    rmse_Y = np.sqrt(mean_squared_error(model_df['Y_true'], model_df['Y_pred']))
    mae_Y = mean_absolute_error(model_df['Y_true'], model_df['Y_pred'])
    rmse_M = np.sqrt(mean_squared_error(model_df['M_true'], model_df['M_pred']))
    mae_M = mean_absolute_error(model_df['M_true'], model_df['M_pred'])

    df_summary = pd.DataFrame([{
        'Model': model_name,
        'CV_RMSE_Y': rmse_Y,
        'CV_MAE_Y': mae_Y,
        'CV_RMSE_M': rmse_M,
        'CV_MAE_M': mae_M,
        'N_HeldOut_Samples': model_df['HeldOut_Sample'].nunique(),
        'N_Predictions': len(model_df),
    }])

    logger.info(f"{model_name} Sample-Out CV: RMSE_Y={rmse_Y:.4f}, MAE_Y={mae_Y:.4f}, "
                f"RMSE_M={rmse_M:.4f}, MAE_M={mae_M:.4f}")

    output_dir = get_output_dir()
    df_results.to_csv(output_dir / 'leave_one_sample_cv_results.csv', index=False, encoding='utf-8')
    df_results.to_excel(output_dir / 'leave_one_sample_cv_results.xlsx', index=False, engine='openpyxl')
    df_summary.to_csv(output_dir / 'leave_one_sample_cv_summary.csv', index=False, encoding='utf-8')
    df_summary.to_excel(output_dir / 'leave_one_sample_cv_summary.xlsx', index=False, engine='openpyxl')

    logger.info(f"Sample-Out CV结果已保存到 {output_dir}")

    return df_results, df_summary
