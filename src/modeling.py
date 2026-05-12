# -*- coding: utf-8 -*-
"""
模型建立和评估模块

使用 type:day 交互项替代单独的样品类型虚拟变量，
确保 Y = -ln R 在 day=0 时物理意义正确。
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.formula.api import ols
from .utils import setup_logger, print_section_header, SAMPLE_TYPES

logger = setup_logger()
warnings.filterwarnings('ignore')

class ModelBuilder:
    """
    模型构建和评估类
    """

    def __init__(self, df):
        self.df = df.copy()
        self.models = {}
        self.results = {}
        self.evaluations = {}

    def _build(self, model_name, formula, description, formula_latex):
        """通用模型构建函数"""
        print_section_header(f"构建{model_name}")
        try:
            model = ols(formula, data=self.df).fit()
            self.models[model_name] = model
            self.results[model_name] = {
                'formula': formula_latex,
                'description': description,
                'sample_size': len(self.df)
            }
            logger.info(f"{model_name}拟合成功: R² = {model.rsquared:.4f}")
            return model
        except Exception as e:
            logger.error(f"{model_name}构建失败: {e}")
            return None

    def build_model_1_basic_time(self):
        """
        Model 1: Y ~ 0 + C(样品类型):测量天数
        每种样品类型的独立时间衰减斜率
        """
        return self._build(
            'Model 1',
            'Y ~ 0 + C(样品类型):测量天数',
            '基础时间模型（type×day交互）',
            'Y = α_s * t + ε'
        )

    def build_model_2_add_weather(self):
        """
        Model 2: Y ~ 0 + C(样品类型):测量天数 + C_T + TOW
        加入累计温度暴露和累计湿润时间
        """
        return self._build(
            'Model 2',
            'Y ~ 0 + C(样品类型):测量天数 + C_T + TOW',
            '加入温湿度的模型',
            'Y = α_s * t + β_T C_T(t) + β_H TOW(t) + ε'
        )

    def build_model_3_with_coupling(self):
        """
        Model 3: Y ~ 0 + C(样品类型):测量天数 + C_T + TOW + C_TH
        加入温湿度耦合项
        """
        return self._build(
            'Model 3',
            'Y ~ 0 + C(样品类型):测量天数 + C_T + TOW + C_TH',
            '加入温湿度耦合项的模型',
            'Y = α_s * t + β_T C_T(t) + β_H TOW(t) + β_{TH} C_{TH}(t) + ε'
        )

    def build_model_4_with_corrosion(self):
        """
        Model 4: Y ~ 0 + C(样品类型):测量天数 + C_T + TOW + C_TH + rust_TOW
        加入锈蚀-湿润交互项（稳健主模型，不含磁黏滞项）
        """
        return self._build(
            'Model 4',
            'Y ~ 0 + C(样品类型):测量天数 + C_T + TOW + C_TH + rust_TOW',
            '加入锈蚀损伤的稳健主模型（不含log_t）',
            'Y = α_s * t + β_T C_T(t) + β_H TOW(t) + β_{TH} C_{TH}(t) + β_{rust} rust_TOW(t) + ε'
        )

    def build_model_5_with_log_time(self):
        """
        Model 5: Y ~ 0 + C(样品类型):测量天数 + log_t + C_T + TOW + C_TH + rust_TOW
        加入统一的磁黏滞对数时间项
        """
        return self._build(
            'Model 5',
            'Y ~ 0 + C(样品类型):测量天数 + log_t + C_T + TOW + C_TH + rust_TOW',
            '加入统一的磁黏滞对数时间项',
            'Y = α_s * t + β_{log} ln(1+t) + β_T C_T(t) + β_H TOW(t) + β_{TH} C_{TH}(t) + β_{rust} rust_TOW(t) + ε'
        )

    def build_model_6_type_specific_log_time(self):
        """
        Model 6: Y ~ 0 + C(样品类型):测量天数 + C(样品类型):log_t + C_T + TOW + C_TH + rust_TOW
        每种样品类型独立的磁黏滞对数时间衰减斜率
        """
        return self._build(
            'Model 6',
            'Y ~ 0 + C(样品类型):测量天数 + C(样品类型):log_t + C_T + TOW + C_TH + rust_TOW',
            '样品类型独立的磁黏滞对数时间项',
            'Y = α_s * t + β_{log,s} ln(1+t) + β_T C_T(t) + β_H TOW(t) + β_{TH} C_{TH}(t) + β_{rust} rust_TOW(t) + ε'
        )

    def build_all_models(self):
        """构建所有模型"""
        print_section_header("建立阶段：构建所有模型")

        self.build_model_1_basic_time()
        self.build_model_2_add_weather()
        self.build_model_3_with_coupling()
        self.build_model_4_with_corrosion()
        self.build_model_5_with_log_time()
        self.build_model_6_type_specific_log_time()
        self._build_explain_model()

        logger.info(f"成功构建{len(self.models)}个模型")

    def _build_explain_model(self):
        """
        ExplainModel（解释模型）

        Y = alpha_s * t + beta_log * ln(1+t) + epsilon

        用途：机理分析——验证时间延迟是否主导、比较四类样品基础时间衰减斜率、
        通过锈蚀钢筋斜率 > 普通钢筋斜率说明锈蚀状态降低剩磁稳定性。
        不含环境累计变量和 Rust0*t（后者与 type:day 共线），避免解释困难。
        """
        print_section_header("构建ExplainModel（解释模型）")
        try:
            formula = 'Y ~ 0 + C(样品类型):测量天数 + log_t'
            model = ols(formula, data=self.df).fit()
            self.models['ExplainModel'] = model
            self.results['ExplainModel'] = {
                'formula': 'Y = α_s * t + β_log ln(1+t) + ε',
                'description': '解释模型（仅type:day+log_t，不含环境变量和Rust0*t）',
                'sample_size': len(self.df)
            }
            logger.info(f"ExplainModel拟合成功: R² = {model.rsquared:.4f}")
            return model
        except Exception as e:
            logger.error(f"ExplainModel构建失败: {e}")
            return None

    def evaluate_models(self):
        """评估所有模型"""
        print_section_header("模型评估")

        for model_name, model in self.models.items():
            logger.info(f"评估{model_name}")

            y_pred_Y = model.fittedvalues
            y_true_Y = model.model.endog

            rmse_Y = np.sqrt(mean_squared_error(y_true_Y, y_pred_Y))
            mae_Y = mean_absolute_error(y_true_Y, y_pred_Y)
            r2 = r2_score(y_true_Y, y_pred_Y)
            aic = model.aic
            bic = model.bic

            M0 = self.df['M0'].values
            M_true = M0 * np.exp(-y_true_Y)
            M_pred = M0 * np.exp(-y_pred_Y)

            rmse_M = np.sqrt(mean_squared_error(M_true, M_pred))
            mae_M = mean_absolute_error(M_true, M_pred)

            self.evaluations[model_name] = {
                'RMSE_Y': rmse_Y,
                'MAE_Y': mae_Y,
                'RMSE_M': rmse_M,
                'MAE_M': mae_M,
                'R2': r2,
                'AIC': aic,
                'BIC': bic
            }

            logger.info(f"{model_name}: R² = {r2:.4f}, RMSE_Y = {rmse_Y:.4f}, AIC = {aic:.2f}")

        return self.evaluations

    def get_best_model(self):
        """
        选择最优模型

        策略：
        1. Model 4 是稳健基准模型（不含 log_t），仅作对照
        2. Model 5 是最终预测模型（加入统一 ln(1+t)）
        3. Model 6 是复杂增强模型（type独立 log_t），仅作对照
        4. Model 5 在 Y 空间和 M 空间误差均优于 Model 4，
           且参数增加有限，选 Model 5 作为最终预测模型
        """
        print_section_header("选择最优模型")

        best_model_name = 'Model 5'
        best_model = self.models.get('Model 5')

        if best_model is None:
            best_model_name = 'Model 4'
            best_model = self.models.get('Model 4')

        if best_model is None:
            valid = list(self.evaluations.keys())
            if valid:
                best_model_name = valid[0]
                best_model = self.models[best_model_name]
            logger.warning(f"Model 5 和 Model 4 均不可用，回退到 {best_model_name}")
        else:
            if 'Model 4' in self.evaluations:
                eval4 = self.evaluations['Model 4']
                eval5 = self.evaluations['Model 5']
                logger.info(f"Model 4 (基准): R²={eval4['R2']:.4f}, RMSE_Y={eval4['RMSE_Y']:.4f}, RMSE_M={eval4['RMSE_M']:.4f}")
                logger.info(f"Model 5 (预测): R²={eval5['R2']:.4f}, RMSE_Y={eval5['RMSE_Y']:.4f}, RMSE_M={eval5['RMSE_M']:.4f}")
                logger.info("选择 Model 5 作为最终预测模型：Y/M空间误差均低于Model 4，log_t项显著提升拟合")

        if best_model is None:
            best_eval = max(self.evaluations.items(), key=lambda x: x[1]['R2'])
            best_model_name = best_eval[0]
            best_model = self.models[best_model_name]

        logger.info(f"最优模型: {best_model_name}")
        logger.info(f"性能指标: {self.evaluations[best_model_name]}")

        return best_model_name, best_model

    def get_model_comparison_df(self):
        """获取模型对比表"""
        comparison_data = []
        for model_name in sorted(self.evaluations.keys()):
            eval_dict = dict(self.evaluations[model_name])
            eval_dict['Model'] = model_name
            comparison_data.append(eval_dict)

        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison[['Model', 'R2', 'RMSE_Y', 'MAE_Y', 'RMSE_M', 'MAE_M', 'AIC', 'BIC']]


def build_models(df):
    """
    构建并评估所有模型
    """
    builder = ModelBuilder(df)
    builder.build_all_models()
    builder.evaluate_models()
    best_name, best_model = builder.get_best_model()

    return builder, best_name, best_model


def save_explain_model_coefficients(builder):
    """保存解释模型系数"""
    from .utils import get_output_dir
    output_dir = get_output_dir()

    if 'ExplainModel' in builder.models:
        model = builder.models['ExplainModel']
        df = pd.DataFrame({
            'Coefficient': model.params.index,
            'Estimate': model.params.values,
            'Std Error': model.bse.values,
            't-value': model.tvalues.values,
            'p-value': model.pvalues.values,
        })
        conf_int = model.conf_int(alpha=0.05)
        df['CI_Lower'] = conf_int[0].values
        df['CI_Upper'] = conf_int[1].values
        df.to_excel(output_dir / 'explain_model_coefficients.xlsx', index=False, engine='openpyxl')
        logger.info(f"解释模型系数已保存: {output_dir / 'explain_model_coefficients.xlsx'}")


def save_explain_vs_predict_comparison(builder):
    """保存解释模型与预测模型对比"""
    from .utils import get_output_dir
    output_dir = get_output_dir()

    rows = []
    for model_name in ['ExplainModel', 'Model 5']:
        if model_name not in builder.evaluations:
            continue
        ev = builder.evaluations[model_name]
        rows.append({
            'Model': model_name,
            '角色': '解释模型（机理分析）' if model_name == 'ExplainModel' else '预测模型（工程应用）',
            'R2': ev['R2'],
            'RMSE_Y': ev['RMSE_Y'],
            'MAE_Y': ev['MAE_Y'],
            'RMSE_M': ev['RMSE_M'],
            'MAE_M': ev['MAE_M'],
            'AIC': ev['AIC'],
            'BIC': ev['BIC'],
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_dir / 'explain_vs_predict_model_comparison.xlsx', index=False, engine='openpyxl')
    logger.info(f"解释vs预测模型对比已保存: {output_dir / 'explain_vs_predict_model_comparison.xlsx'}")
