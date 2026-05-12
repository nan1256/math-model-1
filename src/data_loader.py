# -*- coding: utf-8 -*-
"""
数据加载模块
负责从CSV和Excel文件读取数据，自动识别列名和数据结构
"""

import pandas as pd
import numpy as np
from pathlib import Path
try:
    import chardet
except ImportError:
    chardet = None
from .utils import setup_logger, find_file, print_dataframe_info, print_section_header

logger = setup_logger()

class FieldMapper:
    """
    字段映射器，自动识别原始列名并映射到标准字段名
    """
    
    def __init__(self):
        # 样品类型列的可能名称
        self.sample_type_names = ['样品类型', 'type', '类型', 'sample_type', 'sample', 'specimen_type']
        
        # 样品编号列的可能名称
        self.sample_id_names = ['编号', 'id', '编号', 'sample_id', 'specimen_id', '样品编号']
        
        # 测量天数列的可能名称
        self.day_names = ['测量天数', 'day', 'date', '天数', 't', 'time', 'days']
        
        # 剩磁列的可能名称
        self.remanence_names = ['剩磁(mT)', '剩磁', 'remanence', 'M', '磁性', '磁值', '剩磁值']
        
        # 温度列的可能名称
        self.temperature_names = ['温度(℃)', '温度', 'temperature', 'T', 'temp']
        
        # 湿度列的可能名称
        self.humidity_names = ['相对湿度(%)', '湿度', 'humidity', 'RH', 'relative_humidity', '相对湿度']
        
        # 锈蚀列的可能名称
        self.rust_names = ['锈蚀', '锈', 'rust', 'rust_grade', 'corrosion']
        
    def find_column(self, df, name_list, critical=True):
        """
        在数据框中查找匹配的列
        """
        for col in df.columns:
            if col in name_list:
                return col
            col_lower = str(col).lower()
            for name in name_list:
                name_lower = name.lower()
                if len(name_lower) < 3:
                    continue
                if name_lower in col_lower or col_lower in name_lower:
                    return col
        
        if critical:
            raise ValueError(f"无法找到列: {name_list}。可用列: {df.columns.tolist()}")
        return None
    
    def map_columns(self, df, data_source='experiment'):
        """
        映射数据框的列名
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始数据框
        data_source : str
            数据来源，'experiment'或'weather'
        
        Returns:
        --------
        tuple : (映射后的数据框, 列名映射字典)
        """
        logger.info(f"原始列名: {df.columns.tolist()}")
        
        if data_source == 'experiment':
            # 实验数据映射
            column_map = {}
            
            col_sample_type = self.find_column(df, self.sample_type_names)
            column_map['样品类型'] = col_sample_type
            
            col_sample_id = self.find_column(df, self.sample_id_names)
            column_map['编号'] = col_sample_id
            
            col_day = self.find_column(df, self.day_names)
            column_map['测量天数'] = col_day
            
            col_remanence = self.find_column(df, self.remanence_names)
            column_map['剩磁(mT)'] = col_remanence
            
            col_temperature = self.find_column(df, self.temperature_names, critical=False)
            if col_temperature:
                column_map['温度(℃)'] = col_temperature
            
            col_humidity = self.find_column(df, self.humidity_names, critical=False)
            if col_humidity:
                column_map['相对湿度(%)'] = col_humidity
            
            col_rust = self.find_column(df, self.rust_names, critical=False)
            if col_rust:
                column_map['锈蚀'] = col_rust
            
        elif data_source == 'weather':
            # 天气数据映射
            column_map = {}
            
            col_day = self.find_column(df, self.day_names)
            column_map['天数'] = col_day
            
            col_temperature = self.find_column(df, self.temperature_names)
            column_map['温度'] = col_temperature
            
            col_humidity = self.find_column(df, self.humidity_names)
            column_map['相对湿度'] = col_humidity
        
        else:
            raise ValueError(f"未知的数据来源: {data_source}")
        
        # 反向映射：从原始列名到标准列名
        reverse_map = {v: k for k, v in column_map.items()}
        df_mapped = df.rename(columns=reverse_map)
        
        logger.info(f"列名映射: {column_map}")
        
        return df_mapped, column_map


def load_experiment_data(file_path=None):
    """
    加载实验数据
    
    Parameters:
    -----------
    file_path : str, optional
        文件路径，如果不指定则自动查找
    
    Returns:
    --------
    tuple : (数据框, 列名映射)
    """
    print_section_header("加载实验数据")
    
    if file_path is None:
        # 自动查找文件
        file_path = find_file('附件1_模拟实验数据.csv')
        if file_path is None:
            raise FileNotFoundError("无法找到附件1_模拟实验数据.csv")
    
    file_path = Path(file_path)
    logger.info(f"加载文件: {file_path}")
    
    # 检测编码
    encoding = 'utf-8'
    if chardet is not None:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            encoding_result = chardet.detect(raw_data)
            encoding = encoding_result.get('encoding', 'utf-8')
            logger.info(f"检测到编码: {encoding}")
        except Exception:
            logger.warning("编码检测失败，使用默认编码utf-8")
    else:
        logger.info("chardet未安装，使用默认编码utf-8")
    
    # 读取CSV
    try:
        df = pd.read_csv(file_path, encoding=encoding)
    except:
        # 如果失败，尝试其他编码
        logger.warning(f"使用{encoding}编码失败，尝试gbk")
        df = pd.read_csv(file_path, encoding='gbk')
    
    logger.info(f"数据加载成功，形状: {df.shape}")
    print_dataframe_info(df, "实验数据概览", logger)
    
    # 映射列名
    mapper = FieldMapper()
    df_mapped, column_map = mapper.map_columns(df, data_source='experiment')
    
    return df_mapped, column_map


def load_weather_data(file_path=None):
    """
    加载天气数据
    
    Parameters:
    -----------
    file_path : str, optional
        文件路径，如果不指定则自动查找
    
    Returns:
    --------
    tuple : (数据框, 列名映射)
    """
    print_section_header("加载天气数据")
    
    if file_path is None:
        # 自动查找文件
        file_path = find_file('附件2-weather_data.xlsx')
        if file_path is None:
            logger.warning("未找到weather_data.xlsx，将使用实验数据中的温度和湿度")
            return None, {}
    
    file_path = Path(file_path)
    logger.info(f"加载文件: {file_path}")
    
    # 读取Excel
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        logger.info(f"数据加载成功，形状: {df.shape}")
        print_dataframe_info(df, "天气数据概览", logger)
        
        # 映射列名
        mapper = FieldMapper()
        df_mapped, column_map = mapper.map_columns(df, data_source='weather')
        
        return df_mapped, column_map
    except Exception as e:
        logger.warning(f"加载天气数据失败: {e}")
        return None, {}


def load_all_data(experiment_path=None, weather_path=None):
    """
    加载所有数据
    
    Returns:
    --------
    tuple : (实验数据, 天气数据, 列名映射)
    """
    print_section_header("数据加载阶段")
    
    # 加载实验数据
    df_exp, col_map_exp = load_experiment_data(experiment_path)
    
    # 加载天气数据
    df_weather, col_map_weather = load_weather_data(weather_path)
    
    logger.info("所有数据加载完成")
    
    return df_exp, df_weather, (col_map_exp, col_map_weather)
