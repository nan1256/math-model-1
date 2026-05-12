# -*- coding: utf-8 -*-
"""
特征工程模块
构造环境累计变量
"""

import pandas as pd
import numpy as np
from .utils import setup_logger, print_section_header

logger = setup_logger()

class EnvironmentFeatureEngineer:
    """
    环境特征工程类
    """
    
    def __init__(self, df):
        """
        Parameters:
        -----------
        df : pd.DataFrame
            包含温度和湿度的数据框
        """
        self.df = df.copy()
        self.temp_mean = None
        self.temp_std = None
        self.humidity_mean = None
        self.humidity_std = None
    
    def standardize_environment_variables(self):
        """
        标准化温度和湿度
        
        z_T(d) = [T_d - mean(T)] / std(T)
        z_H(d) = [RH_d - mean(RH)] / std(RH)
        """
        print_section_header("标准化环境变量")
        
        # 获取唯一的天数和对应的温度、湿度
        weather_data = self.df[['测量天数', '温度', '相对湿度']].drop_duplicates()
        weather_data = weather_data.sort_values('测量天数')
        
        # 计算统计量
        self.temp_mean = weather_data['温度'].mean()
        self.temp_std = weather_data['温度'].std()
        self.humidity_mean = weather_data['相对湿度'].mean()
        self.humidity_std = weather_data['相对湿度'].std()
        
        logger.info(f"温度统计: 均值={self.temp_mean:.2f}, 标准差={self.temp_std:.2f}")
        logger.info(f"湿度统计: 均值={self.humidity_mean:.2f}, 标准差={self.humidity_std:.2f}")
        
        # 标准化
        self.df['z_T'] = (self.df['温度'] - self.temp_mean) / (self.temp_std + 1e-6)
        self.df['z_H'] = (self.df['相对湿度'] - self.humidity_mean) / (self.humidity_std + 1e-6)
        
        logger.info("温度和湿度标准化完成")
        
        return self
    
    def compute_cumulative_temperature_exposure(self):
        """
        计算累计温度暴露
        
        C_T(t) = sum_{d=1}^{t} z_T(d)
        """
        print_section_header("计算累计环境变量")
        
        self.df['C_T'] = self.df.groupby('样品ID')['z_T'].cumsum()
        
        logger.info(f"C_T 范围: [{self.df['C_T'].min():.4f}, {self.df['C_T'].max():.4f}]")
        
        return self
    
    def compute_wetness_time(self):
        """
        计算湿润时间
        
        I_d = 1, if RH_d > 80 and T_d > 0
            = 0, otherwise
        
        TOW(t) = sum_{d=1}^{t} I_d
        """
        # 定义湿润指示函数
        self.df['I_wet'] = ((self.df['相对湿度'] > 80) & (self.df['温度'] > 0)).astype(int)
        
        # 计算累计湿润时间
        self.df = self.df.sort_values(['样品ID', '测量天数'])
        self.df['TOW'] = self.df.groupby('样品ID')['I_wet'].cumsum()
        
        logger.info(f"TOW 范围: [{self.df['TOW'].min():.0f}, {self.df['TOW'].max():.0f}]")
        
        return self
    
    def compute_temperature_humidity_coupling(self):
        """
        计算温湿度耦合项
        
        C_TH(t) = sum_{d=1}^{t} z_T(d) * z_H(d)
        """
        # 计算单日耦合值
        self.df['zh_interaction'] = self.df['z_T'] * self.df['z_H']
        
        # 计算累计耦合值
        self.df = self.df.sort_values(['样品ID', '测量天数'])
        self.df['C_TH'] = self.df.groupby('样品ID')['zh_interaction'].cumsum()
        
        logger.info(f"C_TH 范围: [{self.df['C_TH'].min():.4f}, {self.df['C_TH'].max():.4f}]")
        
        return self
    
    def compute_alternative_coupling(self):
        """
        计算备选的温湿度耦合项（针对高湿腐蚀）
        
        C_TH2(t) = sum_{d=1}^{t} z_T(d) * max(0, RH_d - 80)
        """
        # 高湿指示
        self.df['high_humidity_excess'] = np.maximum(0, self.df['相对湿度'] - 80)
        
        # 单日值
        self.df['zh_interaction2'] = self.df['z_T'] * self.df['high_humidity_excess']
        
        # 累计值
        self.df = self.df.sort_values(['样品ID', '测量天数'])
        self.df['C_TH2'] = self.df.groupby('样品ID')['zh_interaction2'].cumsum()
        
        logger.info(f"C_TH2 范围: [{self.df['C_TH2'].min():.4f}, {self.df['C_TH2'].max():.4f}]")
        
        return self
    
    def compute_corrosion_damage_index(self):
        """
        计算锈蚀损伤交互项

        rust0 = rust_grade (0 for non-rusted, 3 for rusted rebar, treated as continuous)
        rust_TOW = rust0 * TOW
        rust_day = rust0 * day
        rust_log_t = rust0 * log_t
        
        优先使用 rust_TOW 作为主锈蚀损伤项
        """
        self.df['rust0'] = self.df['锈蚀等级'].astype(float)
        self.df['rust_TOW'] = self.df['rust0'] * self.df['TOW']
        self.df['rust_day'] = self.df['rust0'] * self.df['测量天数']
        
        if 'log_t' not in self.df.columns:
            self.df['log_t'] = np.log(1 + self.df['测量天数'])
        self.df['rust_log_t'] = self.df['rust0'] * self.df['log_t']
        
        logger.info(f"rust0 范围: [{self.df['rust0'].min():.0f}, {self.df['rust0'].max():.0f}]")
        logger.info(f"rust_TOW 范围: [{self.df['rust_TOW'].min():.0f}, {self.df['rust_TOW'].max():.0f}]")
        
        return self
    
    def compute_log_time_feature(self):
        """
        计算磁黏滞对数时间项
        
        log_t = ln(1 + t)
        """
        self.df['log_t'] = np.log(1 + self.df['测量天数'])
        
        logger.info(f"log_t 范围: [{self.df['log_t'].min():.4f}, {self.df['log_t'].max():.4f}]")
        
        return self
    
    def get_engineered_data(self):
        """
        获取特征工程后的数据
        """
        return self.df.copy()


def engineer_features(df):
    """
    执行完整的特征工程
    
    Parameters:
    -----------
    df : pd.DataFrame
        预处理后的数据
    
    Returns:
    --------
    pd.DataFrame
        特征工程后的数据
    """
    print_section_header("特征工程阶段")
    
    engineer = EnvironmentFeatureEngineer(df)
    
    # 执行特征工程步骤
    engineer.standardize_environment_variables()
    engineer.compute_cumulative_temperature_exposure()
    engineer.compute_wetness_time()
    engineer.compute_temperature_humidity_coupling()
    engineer.compute_alternative_coupling()
    engineer.compute_corrosion_damage_index()
    engineer.compute_log_time_feature()
    
    df_engineered = engineer.get_engineered_data()
    
    logger.info("特征工程完成")
    logger.info(f"最终数据列数: {df_engineered.shape[1]}")
    logger.info(f"最终数据行数: {df_engineered.shape[0]}")
    
    return df_engineered, engineer
