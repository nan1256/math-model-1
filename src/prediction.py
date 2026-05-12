# -*- coding: utf-8 -*-
"""
预测模块
使用 patsy 构建设计矩阵进行预测
"""

import pandas as pd
import numpy as np
from .utils import setup_logger, print_section_header, get_static_threshold

logger = setup_logger()

class Predictor:
    """
    预测类
    """

    def __init__(self, df, best_model, engineer):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            训练数据
        best_model :
            最优的OLS模型
        engineer : EnvironmentFeatureEngineer
            特征工程器
        """
        self.df = df
        self.best_model = best_model
        self.engineer = engineer
        self.model_params = best_model.params
        self.model_summary = best_model.summary()
        self.residuals = best_model.resid

    def predict_days_23_29(self, df_full_weather):
        """
        预测第23-29天的剩磁值（含预测区间）
        """
        print_section_header("预测第23-29天的剩磁值")

        prediction_days = list(range(23, 30))
        residual_std = np.std(self.residuals)

        sample_type_M0 = self.df.groupby('样品类型')['M0'].mean()
        logger.info(f"各样品类型平均初始剩磁:")
        logger.info(sample_type_M0)

        prediction_results = []

        for day in prediction_days:
            result_row = {'day': day}

            for sample_type in ['小号铁钉', '小号铁夹', '普通钢筋', '锈蚀钢筋']:
                weather_row = df_full_weather[df_full_weather['测量天数'] == day]

                if weather_row.empty:
                    logger.warning(f"缺少第{day}天的天气数据")
                    for suf in ['', '_lower', '_upper']:
                        result_row[sample_type + suf] = np.nan
                else:
                    row = weather_row.iloc[0].to_dict()
                    row['样品类型'] = sample_type
                    row['测量天数'] = day

                    df_pred = pd.DataFrame([row])
                    Y_pred = self.best_model.predict(df_pred)[0]

                    M0_mean = sample_type_M0[sample_type]
                    M_pred = M0_mean * np.exp(-Y_pred)
                    M_lower = M0_mean * np.exp(-(Y_pred + 1.96 * residual_std))
                    M_upper = M0_mean * np.exp(-(Y_pred - 1.96 * residual_std))

                    result_row[sample_type] = M_pred
                    result_row[f'{sample_type}_lower'] = M_lower
                    result_row[f'{sample_type}_upper'] = M_upper

            prediction_results.append(result_row)

        df_predictions = pd.DataFrame(prediction_results)

        logger.info(f"预测完成，形状: {df_predictions.shape}")
        logger.info(f"\n{df_predictions[['day'] + [s for s in ['小号铁钉','小号铁夹','普通钢筋','锈蚀钢筋']]]}")

        return df_predictions

    def predict_dynamic_thresholds_1_90(self, df_full_weather):
        """
        预测第1-90天的动态阈值

        T_dyn,s(t) = T_0,s * exp[-Y_hat_s(t)]

        从 day=1 开始生成动态阈值（day=0 时 Y=0 精确成立）
        """
        print_section_header("预测1-90天的动态阈值")

        threshold_results = []

        for day in range(1, 91):
            result_row = {'day': day}

            weather_row = df_full_weather[df_full_weather['测量天数'] == day]

            if weather_row.empty:
                logger.warning(f"缺少第{day}天的天气数据，使用线性插值")
                for sample_type in ['小号铁钉', '小号铁夹', '普通钢筋', '锈蚀钢筋']:
                    result_row[f'{sample_type}_阈值'] = np.nan
            else:
                for sample_type in ['小号铁钉', '小号铁夹', '普通钢筋', '锈蚀钢筋']:
                    row = weather_row.iloc[0].to_dict()
                    row['样品类型'] = sample_type
                    row['测量天数'] = day

                    df_pred = pd.DataFrame([row])
                    Y_pred = self.best_model.predict(df_pred)[0]

                    T_static = get_static_threshold(sample_type)
                    T_dyn = T_static * np.exp(-Y_pred)

                    result_row[f'{sample_type}_阈值'] = T_dyn

            threshold_results.append(result_row)

        df_thresholds = pd.DataFrame(threshold_results)

        for col in df_thresholds.columns:
            if col != 'day':
                df_thresholds[col] = df_thresholds[col].interpolate(method='linear')

        logger.info(f"动态阈值预测完成，形状: {df_thresholds.shape}")

        return df_thresholds

    def get_model_summary(self):
        return self.model_summary

    def get_residuals(self):
        return self.residuals

    def get_coefficients_summary(self):
        """获取系数摘要表"""
        summary_table = pd.DataFrame({
            'Coefficient': self.model_params.index,
            'Estimate': self.model_params.values,
            'Std Error': self.best_model.bse.values,
            't-value': self.best_model.tvalues.values,
            'p-value': self.best_model.pvalues.values
        })

        conf_int = self.best_model.conf_int(alpha=0.05)
        summary_table['CI_Lower'] = conf_int[0].values
        summary_table['CI_Upper'] = conf_int[1].values

        return summary_table


def predict(df, best_model, engineer, df_weather_full):
    """
    执行预测

    Returns:
    --------
    tuple : (预测器, 第23-29天预测结果, 1-90天动态阈值)
    """
    print_section_header("预测阶段")

    predictor = Predictor(df, best_model, engineer)

    df_pred_23_29 = predictor.predict_days_23_29(df_weather_full)
    df_thresholds = predictor.predict_dynamic_thresholds_1_90(df_weather_full)

    return predictor, df_pred_23_29, df_thresholds
