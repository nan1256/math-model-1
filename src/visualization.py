# -*- coding: utf-8 -*-
"""
可视化模块
生成各种图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from .utils import setup_logger, print_section_header, get_output_dir, SAMPLE_TYPES, get_static_threshold

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

logger = setup_logger()

class Visualizer:
    """
    可视化类
    """
    
    def __init__(self, df, predictor, df_pred_23_29, df_thresholds):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            训练数据（带所有特征）
        predictor : Predictor
            预测器对象
        df_pred_23_29 : pd.DataFrame
            23-29天预测结果
        df_thresholds : pd.DataFrame
            1-90天动态阈值
        """
        self.df = df
        self.predictor = predictor
        self.df_pred_23_29 = df_pred_23_29
        self.df_thresholds = df_thresholds
        
        self.output_dir = get_output_dir() / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"图表保存目录: {self.output_dir}")
    
    def plot_average_remanence_by_type(self):
        """
        图1：四类样品平均剩磁随时间变化曲线
        """
        logger.info("绘制图1：平均剩磁随时间变化")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for sample_type in SAMPLE_TYPES:
            data = self.df[self.df['样品类型'] == sample_type]
            avg_by_day = data.groupby('测量天数')['剩磁(mT)'].mean()
            
            ax.plot(avg_by_day.index, avg_by_day.values, marker='o', 
                   label=sample_type, linewidth=2, markersize=6)
        
        ax.set_xlabel('测量天数 (天)', fontsize=12, fontweight='bold')
        ax.set_ylabel('平均剩磁值 (mT)', fontsize=12, fontweight='bold')
        ax.set_title('四类样品平均剩磁随时间变化曲线', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / '01_average_remanence.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()
    
    def plot_remanence_ratio_by_type(self):
        """
        图2：四类样品平均剩磁保持率 R(t) 随时间变化曲线
        """
        logger.info("绘制图2：平均保持率随时间变化")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for sample_type in SAMPLE_TYPES:
            data = self.df[self.df['样品类型'] == sample_type]
            avg_by_day = data.groupby('测量天数')['R'].mean()
            
            ax.plot(avg_by_day.index, avg_by_day.values, marker='o', 
                   label=sample_type, linewidth=2, markersize=6)
        
        ax.set_xlabel('测量天数 (天)', fontsize=12, fontweight='bold')
        ax.set_ylabel('平均剩磁保持率 R(t)', fontsize=12, fontweight='bold')
        ax.set_title('四类样品平均剩磁保持率随时间变化曲线', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        output_path = self.output_dir / '02_remanence_ratio.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()
    
    def plot_temperature_humidity(self):
        """
        图3：温度和湿度随时间变化曲线
        """
        logger.info("绘制图3：温度和湿度变化")
        
        # 获取唯一的天数和对应的温湿度
        weather_data = self.df[['测量天数', '温度', '相对湿度']].drop_duplicates()
        weather_data = weather_data.sort_values('测量天数')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
        
        # 温度曲线
        ax1.plot(weather_data['测量天数'], weather_data['温度'], 
                marker='o', color='red', linewidth=2, markersize=5)
        ax1.set_xlabel('测量天数 (天)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('温度 (℃)', fontsize=11, fontweight='bold')
        ax1.set_title('环境温度随时间变化', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 湿度曲线
        ax2.plot(weather_data['测量天数'], weather_data['相对湿度'], 
                marker='s', color='blue', linewidth=2, markersize=5)
        ax2.axhline(y=80, color='green', linestyle='--', linewidth=2, label='湿润阈值 (80%)')
        ax2.set_xlabel('测量天数 (天)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('相对湿度 (%)', fontsize=11, fontweight='bold')
        ax2.set_title('环境相对湿度随时间变化', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        output_path = self.output_dir / '03_temperature_humidity.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()
    
    def plot_cumulative_environment_features(self):
        """
        图4：TOW(t)、C_T(t)、C_TH(t) 的累计变化曲线
        """
        logger.info("绘制图4：累计环境特征变化")
        
        # 取第一个样品作为代表（或平均值）
        weather_data = self.df[['测量天数', 'C_T', 'TOW', 'C_TH']].drop_duplicates()
        weather_data = weather_data.sort_values('测量天数')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # C_T
        axes[0].plot(weather_data['测量天数'], weather_data['C_T'], 
                    marker='o', color='red', linewidth=2)
        axes[0].set_xlabel('测量天数 (天)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('C_T (累计温度暴露)', fontsize=11, fontweight='bold')
        axes[0].set_title('累计温度暴露变化', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # TOW
        axes[1].plot(weather_data['测量天数'], weather_data['TOW'], 
                    marker='s', color='blue', linewidth=2)
        axes[1].set_xlabel('测量天数 (天)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('TOW (累计湿润时间)', fontsize=11, fontweight='bold')
        axes[1].set_title('累计湿润时间变化', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # C_TH
        axes[2].plot(weather_data['测量天数'], weather_data['C_TH'], 
                    marker='^', color='green', linewidth=2)
        axes[2].set_xlabel('测量天数 (天)', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('C_TH (温湿度耦合)', fontsize=11, fontweight='bold')
        axes[2].set_title('温湿度耦合项变化', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        output_path = self.output_dir / '04_cumulative_features.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()
    
    def plot_measured_vs_predicted(self):
        """
        图5：实测值 vs 预测值散点图
        """
        logger.info("绘制图5：实测值vs预测值")
        
        # 获取预测值
        y_pred_Y = self.predictor.best_model.fittedvalues
        y_true_Y = self.predictor.best_model.model.endog
        
        M0 = self.df['M0'].values
        M_true = M0 * np.exp(-y_true_Y)
        M_pred = M0 * np.exp(-y_pred_Y)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(M_true, M_pred, alpha=0.6, s=50)
        
        # 添加完美预测线
        min_val = min(M_true.min(), M_pred.min())
        max_val = max(M_true.max(), M_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
        
        ax.set_xlabel('实测剩磁值 (mT)', fontsize=12, fontweight='bold')
        ax.set_ylabel('预测剩磁值 (mT)', fontsize=12, fontweight='bold')
        ax.set_title('模型预测值与实测值对比', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / '05_measured_vs_predicted.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()
    
    def plot_residuals_distribution(self):
        """
        图6：残差分布图
        """
        logger.info("绘制图6：残差分布")
        
        residuals = self.predictor.get_residuals()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 残差直方图
        axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('残差值', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('频数', fontsize=11, fontweight='bold')
        axes[0].set_title('残差分布直方图', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q图（正态性检验）', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        output_path = self.output_dir / '06_residuals_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()
    
    def plot_prediction_23_29(self):
        """
        图7：四类样品第 23–29 天预测剩磁曲线
        """
        logger.info("绘制图7：23-29天预测结果")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for sample_type in SAMPLE_TYPES:
            ax.plot(self.df_pred_23_29['day'], 
                   self.df_pred_23_29[sample_type], 
                   marker='o', label=sample_type, linewidth=2, markersize=6)
        
        ax.set_xlabel('天数 (天)', fontsize=12, fontweight='bold')
        ax.set_ylabel('预测剩磁值 (mT)', fontsize=12, fontweight='bold')
        ax.set_title('四类样品第23-29天预测剩磁曲线', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(self.df_pred_23_29['day'])
        
        output_path = self.output_dir / '07_prediction_23_29.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()
    
    def plot_dynamic_thresholds(self):
        """
        图8：四类样品 1–90 天动态阈值曲线
        """
        logger.info("绘制图8：1-90天动态阈值")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for sample_type in SAMPLE_TYPES:
            col_name = f'{sample_type}_阈值'
            if col_name in self.df_thresholds.columns:
                ax.plot(self.df_thresholds['day'], 
                       self.df_thresholds[col_name], 
                       marker='o', label=sample_type, linewidth=2, markersize=3)
        
        ax.set_xlabel('天数 (天)', fontsize=12, fontweight='bold')
        ax.set_ylabel('动态阈值 (mT)', fontsize=12, fontweight='bold')
        ax.set_title('四类样品1-90天动态阈值变化曲线', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / '08_dynamic_thresholds.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()
    
    def plot_corrosion_comparison(self):
        """
        图9：普通钢筋 vs 锈蚀钢筋衰减对比图
        """
        logger.info("绘制图9：普通钢筋vs锈蚀钢筋对比")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for sample_type in ['普通钢筋', '锈蚀钢筋']:
            data = self.df[self.df['样品类型'] == sample_type]
            avg_by_day = data.groupby('测量天数')['Y'].mean()
            
            ax.plot(avg_by_day.index, avg_by_day.values, marker='o', 
                   label=sample_type, linewidth=2, markersize=7)
        
        ax.set_xlabel('测量天数 (天)', fontsize=12, fontweight='bold')
        ax.set_ylabel('衰减强度 Y(t) = -ln R(t)', fontsize=12, fontweight='bold')
        ax.set_title('普通钢筋与锈蚀钢筋衰减速率对比', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / '09_corrosion_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()
    
    def plot_cv_comparison(self, df_cv_summary):
        """
        图11：Leave-One-Day-Out CV 模型对比图
        """
        logger.info("绘制图11：CV模型对比")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        models = df_cv_summary['Model'].tolist()
        x = np.arange(len(models))
        width = 0.35

        ax1 = axes[0]
        bars1 = ax1.bar(x - width/2, df_cv_summary['CV_RMSE_Y'].values, width, label='RMSE_Y', color='steelblue')
        bars2 = ax1.bar(x + width/2, df_cv_summary['CV_MAE_Y'].values, width, label='MAE_Y', color='coral')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=10)
        ax1.set_ylabel('Y空间误差', fontsize=11, fontweight='bold')
        ax1.set_title('Leave-One-Day-Out CV (Y空间)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[1]
        bars3 = ax2.bar(x - width/2, df_cv_summary['CV_RMSE_M'].values, width, label='RMSE_M (mT)', color='steelblue')
        bars4 = ax2.bar(x + width/2, df_cv_summary['CV_MAE_M'].values, width, label='MAE_M (mT)', color='coral')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, fontsize=10)
        ax2.set_ylabel('M空间误差 (mT)', fontsize=11, fontweight='bold')
        ax2.set_title('Leave-One-Day-Out CV (M空间)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar in [bars1, bars2, bars3, bars4]:
            for rect in bar:
                h = rect.get_height()
                ax = rect.axes
                ax.text(rect.get_x() + rect.get_width()/2., h + 0.002,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        output_path = self.output_dir / '11_leave_one_day_cv_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()

    def plot_monotonic_threshold_comparison(self, df_original, df_monotonic):
        """
        图12：原始阈值 vs 单调修正阈值对比
        """
        logger.info("绘制图12：单调修正阈值对比")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, sample_type in enumerate(SAMPLE_TYPES[:4]):
            ax = axes[idx]
            col = f'{sample_type}_阈值'
            if col not in df_original.columns:
                continue

            ax.plot(df_original['day'], df_original[col],
                    marker='o', markersize=2, linewidth=1.5, alpha=0.6,
                    color='steelblue', label='原始阈值')
            ax.plot(df_monotonic['day'], df_monotonic[col],
                    marker='', linewidth=2, color='red', label='单调修正')

            T0 = get_static_threshold(sample_type)
            ax.axhline(y=T0, color='gray', linestyle=':', linewidth=1.5, label=f'静态阈值 T0={T0}')

            ax.set_xlabel('天数', fontsize=10, fontweight='bold')
            ax.set_ylabel('阈值 (mT)', fontsize=10, fontweight='bold')
            ax.set_title(f'{sample_type}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle('原始动态阈值 vs 单调修正动态阈值', fontsize=14, fontweight='bold')
        fig.tight_layout()

        output_path = self.output_dir / '12_monotonic_threshold_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()

    def plot_sample_cv_comparison(self, df_sample_cv_summary):
        """图13：Leave-One-Sample-Out CV 模型对比"""
        logger.info("绘制图13：Sample-Out CV")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        models = df_sample_cv_summary['Model'].tolist()
        x = np.arange(len(models))
        width = 0.35

        ax1 = axes[0]
        bars1 = ax1.bar(x - width/2, df_sample_cv_summary['CV_RMSE_Y'].values, width,
                        label='RMSE_Y', color='steelblue')
        bars2 = ax1.bar(x + width/2, df_sample_cv_summary['CV_MAE_Y'].values, width,
                        label='MAE_Y', color='coral')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=10)
        ax1.set_ylabel('Y空间误差', fontsize=11, fontweight='bold')
        ax1.set_title('Leave-One-Sample-Out CV (Y空间)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        for bar in [bars1, bars2]:
            for rect in bar:
                h = rect.get_height()
                rect.axes.text(rect.get_x() + rect.get_width()/2., h + 0.002,
                               f'{h:.3f}', ha='center', va='bottom', fontsize=7)

        ax2 = axes[1]
        bars3 = ax2.bar(x - width/2, df_sample_cv_summary['CV_RMSE_M'].values, width,
                        label='RMSE_M (mT)', color='steelblue')
        bars4 = ax2.bar(x + width/2, df_sample_cv_summary['CV_MAE_M'].values, width,
                        label='MAE_M (mT)', color='coral')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, fontsize=10)
        ax2.set_ylabel('M空间误差 (mT)', fontsize=11, fontweight='bold')
        ax2.set_title('Leave-One-Sample-Out CV (M空间)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        for bar in [bars3, bars4]:
            for rect in bar:
                h = rect.get_height()
                rect.axes.text(rect.get_x() + rect.get_width()/2., h + 0.002,
                               f'{h:.3f}', ha='center', va='bottom', fontsize=7)

        fig.tight_layout()
        output_path = self.output_dir / '13_leave_one_sample_cv_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()

    def plot_dynamic_threshold_interval(self, df_interval):
        """图14：动态阈值置信区间"""
        logger.info("绘制图14：动态阈值置信区间")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, sample_type in enumerate(SAMPLE_TYPES[:4]):
            ax = axes[idx]
            data = df_interval[df_interval['样品类型'] == sample_type]

            ax.fill_between(data['day'], data['threshold_lower_95'], data['threshold_upper_95'],
                            alpha=0.2, color='steelblue', label='95% 置信区间')
            ax.plot(data['day'], data['threshold_mean'], linewidth=2, color='steelblue', label='均值')

            if 'threshold_mono_mean' in data.columns:
                ax.plot(data['day'], data['threshold_mono_mean'],
                        linewidth=2, color='red', linestyle='--', label='单调修正')

            T0 = get_static_threshold(sample_type)
            ax.axhline(y=T0, color='gray', linestyle=':', linewidth=1.5, label=f'T0={T0}')

            ax.set_xlabel('天数', fontsize=10, fontweight='bold')
            ax.set_ylabel('阈值 (mT)', fontsize=10, fontweight='bold')
            ax.set_title(f'{sample_type}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle('动态阈值及95%置信区间', fontsize=14, fontweight='bold')
        fig.tight_layout()
        output_path = self.output_dir / '14_dynamic_threshold_interval.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()

    def plot_decision_rule_diagram(self):
        """图15：雷击判定规则示意图"""
        logger.info("绘制图15：雷击判定规则")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        rules = [
            (9.0, '高置信雷击', '#d9534f', 'M_obs ≥ T_upper_95'),
            (7.0, '疑似雷击', '#f0ad4e', 'T_mean ≤ M_obs < T_upper_95'),
            (5.0, '低置信疑似', '#5bc0de', 'T_lower_95 ≤ M_obs < T_mean'),
            (3.0, '雷击证据不足', '#5cb85c', 'M_obs < T_lower_95'),
        ]

        for y, label, color, desc in rules:
            ax.barh(y, 8, 1.2, left=1, color=color, alpha=0.8, edgecolor='black')
            ax.text(1.2, y, label, va='center', fontsize=12, fontweight='bold', color='white')
            ax.text(5.5, y, desc, va='center', fontsize=10, color='black')

        ax.text(5, 9.8, '雷击判定分级规则', ha='center', fontsize=16, fontweight='bold')
        ax.text(5, 1.2, '箭头方向：剩磁值降低 →', ha='center', fontsize=9, color='gray')

        ax.annotate('', xy=(0.3, 9), xytext=(0.3, 3),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        fig.tight_layout()
        output_path = self.output_dir / '15_decision_rule_diagram.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"已保存: {output_path}")
        plt.close()

    def generate_all_plots(self, df_cv_summary=None, df_monotonic=None, df_original_thresholds=None,
                           df_sample_cv_summary=None, df_interval=None):
        """
        生成所有图表
        """
        print_section_header("绘图阶段")

        plots = [
            (self.plot_average_remanence_by_type, "图1"),
            (self.plot_remanence_ratio_by_type, "图2"),
            (self.plot_temperature_humidity, "图3"),
            (self.plot_cumulative_environment_features, "图4"),
            (self.plot_measured_vs_predicted, "图5"),
            (self.plot_residuals_distribution, "图6"),
            (self.plot_prediction_23_29, "图7"),
            (self.plot_dynamic_thresholds, "图8"),
            (self.plot_corrosion_comparison, "图9"),
        ]

        for plot_func, name in plots:
            try:
                plot_func()
            except Exception as e:
                logger.error(f"绘制{name}失败: {e}")

        if df_cv_summary is not None:
            try:
                self.plot_cv_comparison(df_cv_summary)
            except Exception as e:
                logger.error(f"绘制图11失败: {e}")

        if df_monotonic is not None and df_original_thresholds is not None:
            try:
                self.plot_monotonic_threshold_comparison(df_original_thresholds, df_monotonic)
            except Exception as e:
                logger.error(f"绘制图12失败: {e}")

        if df_sample_cv_summary is not None:
            try:
                self.plot_sample_cv_comparison(df_sample_cv_summary)
            except Exception as e:
                logger.error(f"绘制图13失败: {e}")

        if df_interval is not None:
            try:
                self.plot_dynamic_threshold_interval(df_interval)
            except Exception as e:
                logger.error(f"绘制图14失败: {e}")

        try:
            self.plot_decision_rule_diagram()
        except Exception as e:
            logger.error(f"绘制图15失败: {e}")

        logger.info("图表生成完成")

def visualize(df, predictor, df_pred_23_29, df_thresholds, df_cv_summary=None, df_monotonic=None,
              df_sample_cv_summary=None, df_interval=None):
    """
    执行可视化
    """
    visualizer = Visualizer(df, predictor, df_pred_23_29, df_thresholds)
    visualizer.generate_all_plots(df_cv_summary=df_cv_summary,
                                   df_monotonic=df_monotonic,
                                   df_original_thresholds=df_thresholds,
                                   df_sample_cv_summary=df_sample_cv_summary,
                                   df_interval=df_interval)
    return visualizer
