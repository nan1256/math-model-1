# -*- coding: utf-8 -*-
"""
工具函数模块
提供公共的日志、路径和数据处理工具
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# 设置日志
def setup_logger(log_file=None):
    """
    配置日志系统
    """
    logger = logging.getLogger('ModelingEngine')
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 控制台处理器
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    ch.setFormatter(formatter)
    ch.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))
    logger.addHandler(ch)
    
    # 文件处理器（如果指定）
    if log_file:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def get_project_root():
    """
    获取项目根目录
    """
    return Path(__file__).parent.parent

def get_data_dir():
    """
    获取数据目录
    """
    root = get_project_root()
    data_dir = root / 'data'
    if not data_dir.exists():
        # 如果data目录不存在，检查是否直接在项目根目录
        if (root / '附件1_模拟实验数据.csv').exists():
            return root
        else:
            # 创建data目录
            data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_output_dir():
    """
    获取输出目录
    """
    root = get_project_root()
    output_dir = root / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def find_file(filename, search_paths=None):
    """
    查找文件，支持多个搜索路径
    """
    if search_paths is None:
        search_paths = [get_data_dir(), get_project_root()]
    
    for path in search_paths:
        file_path = Path(path) / filename
        if file_path.exists():
            return file_path
    
    return None

def detect_encoding(file_path):
    """
    自动检测文件编码
    """
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        result = chardet.detect(raw_data)
        return result.get('encoding', 'utf-8')
    except ImportError:
        return 'utf-8'

def ensure_dir_exists(dir_path):
    """
    确保目录存在
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def print_section_header(title, logger=None):
    """
    打印章节头
    """
    header = f"\n{'='*60}\n{title}\n{'='*60}"
    if logger:
        logger.info(header)
    else:
        print(header)

def print_dataframe_info(df, title, logger=None):
    """
    打印数据框信息
    """
    msg = f"\n{title}:\n"
    msg += f"  形状: {df.shape}\n"
    msg += f"  列数: {df.columns.tolist()}\n"
    msg += f"  缺失值:\n{df.isnull().sum()}\n"
    msg += f"  数据类型:\n{df.dtypes}\n"
    
    if logger:
        logger.info(msg)
    else:
        print(msg)

# 样品类型和初始阈值映射
SAMPLE_TYPE_CONFIG = {
    '小号铁钉': {'initial_threshold': 1.0, 'rust_grade': 0},
    '小号铁夹': {'initial_threshold': 1.0, 'rust_grade': 0},
    '普通钢筋': {'initial_threshold': 1.5, 'rust_grade': 0},
    '锈蚀钢筋': {'initial_threshold': 1.5, 'rust_grade': 3},
}

SAMPLE_TYPES = ['小号铁钉', '小号铁夹', '普通钢筋', '锈蚀钢筋']

def get_static_threshold(sample_type):
    """
    获取样品类型的静态阈值
    """
    if sample_type in SAMPLE_TYPE_CONFIG:
        return SAMPLE_TYPE_CONFIG[sample_type]['initial_threshold']
    return 1.0

def get_rust_grade(sample_type):
    """
    获取样品类型的锈蚀等级
    """
    if sample_type in SAMPLE_TYPE_CONFIG:
        return SAMPLE_TYPE_CONFIG[sample_type]['rust_grade']
    return 0
