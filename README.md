# 剩磁法雷击判定预测 - 数学建模工程

重庆邮电大学数学建模竞赛 B题：剩磁法雷击判定预测

## 项目概述

本项目基于环境累计暴露驱动的剩磁衰减模型，完成了从数据加载、预处理、特征工程、模型建立、预测到可视化的完整数学建模工程。

### 核心创新

1. **剩磁保持率建模**：不直接拟合原始剩磁，而是通过保持率 $R(t) = M(t)/M(0)$ 的衰减强度 $Y(t) = -\ln R(t)$ 进行建模。

2. **环境累计变量**：构造了温度累计暴露、湿润时间、温湿度耦合等环境特征，更好地捕捉环境对剩磁衰减的影响。

3. **锈蚀效应建模**：将锈蚀状态作为动态损伤指数纳入模型，区分了不同锈蚀程度样品的衰减差异。

4. **动态阈值修正**：基于模型预测的衰减趋势，动态调整雷击判定阈值，提高判定准确性。

## 项目结构

```
math model/
├── data/                        # 数据目录
│   ├── 附件1_模拟实验数据.csv   # 实验数据
│   └── 附件2-weather_data.xlsx   # 天气数据
│
├── src/                         # 源代码模块
│   ├── __init__.py             # 包初始化
│   ├── utils.py                # 工具函数
│   ├── data_loader.py          # 数据加载
│   ├── preprocessing.py        # 数据预处理
│   ├── feature_engineering.py  # 特征工程
│   ├── modeling.py             # 模型建立
│   ├── prediction.py           # 预测模块
│   └── visualization.py        # 可视化模块
│
├── outputs/                     # 输出结果
│   ├── figures/                # 12张图表输出
│   ├── model_comparison.xlsx   # 模型对比表
│   ├── main_model_coefficients.xlsx    # 模型参数表
│   ├── prediction_23_29.xlsx   # 第23-29天预测（含区间）
│   ├── dynamic_threshold_1_90.xlsx     # 1-90天动态阈值
│   ├── monotonic_dynamic_threshold_1_90.xlsx  # 单调修正阈值
│   ├── leave_one_day_cv_results.xlsx   # CV详细结果
│   ├── leave_one_day_cv_summary.xlsx   # CV汇总
│   ├── feature_correlation_matrix.xlsx  # 相关系数矩阵
│   ├── feature_vif.xlsx        # VIF表
│   ├── processed_data.csv      # 处理后的数据
│   ├── main_model_summary.txt  # 模型摘要
│   └── modeling_summary.md     # 论文摘要
│
├── main.py                      # 主程序
├── requirements.txt            # Python依赖
├── README.md                   # 本文件
└── .gitignore                 # Git忽略文件
```

## 快速开始

### 1. 环境配置

#### 使用 pip 安装

```bash
# 进入项目目录
cd "path/to/math model"

# 创建虚拟环境（可选但推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 使用 conda 安装

```bash
# 创建新环境
conda create -n modeling python=3.9

# 激活环境
conda activate modeling

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行主程序

```bash
# 运行完整建模流程
python main.py
```

程序将自动执行以下步骤：
1. ✓ 加载并识别列名
2. ✓ 数据预处理和清洗
3. ✓ 特征工程（构造累计环境变量）
4. ✓ 构建6个模型并比较
5. ✓ 选择最终预测模型 (Model 5)
6. ✓ 预测第23-29天的剩磁值
7. ✓ 预测1-90天的动态阈值
8. ✓ 生成12张专业图表
9. ✓ 输出Excel和CSV结果

### 3. 查看结果

#### 模型对比结果
```
outputs/model_comparison.xlsx
```

#### 主模型参数
```
outputs/main_model_coefficients.xlsx
```

#### 预测结果
```
outputs/prediction_23_29.xlsx        # 第23-29天预测（含区间）
outputs/dynamic_threshold_1_90.xlsx  # 1-90天动态阈值
outputs/monotonic_dynamic_threshold_1_90.xlsx  # 单调修正阈值
```

#### 验证与诊断
```
outputs/leave_one_day_cv_results.xlsx   # Leave-One-Day-Out CV
outputs/leave_one_day_cv_summary.xlsx   # CV汇总
outputs/feature_correlation_matrix.xlsx  # 相关系数矩阵
outputs/feature_vif.xlsx        # VIF共线性诊断
```

#### 可视化图表
```
outputs/figures/01_average_remanence.png
outputs/figures/02_remanence_ratio.png
outputs/figures/03_temperature_humidity.png
outputs/figures/04_cumulative_features.png
outputs/figures/05_measured_vs_predicted.png
outputs/figures/06_residuals_distribution.png
outputs/figures/07_prediction_23_29.png
outputs/figures/08_dynamic_thresholds.png
outputs/figures/09_corrosion_comparison.png
outputs/figures/10_feature_correlation_heatmap.png
outputs/figures/11_leave_one_day_cv_comparison.png
outputs/figures/12_monotonic_threshold_comparison.png
```

#### 论文摘要
```
outputs/modeling_summary.md
```

## 数据文件说明

### 附件1：模拟实验数据.csv

必须包含以下列（自动识别，列名可以是中文或英文）：

| 列名 | 说明 | 示例 |
|------|------|------|
| 样品类型 | 样品的类型 | 小号铁钉、小号铁夹、普通钢筋、锈蚀钢筋 |
| 编号 | 同类样品的编号 | 1, 2, 3, ... |
| 测量天数 | 测量的天数 | 0, 1, 2, ..., 90 |
| 剩磁(mT) | 测量时刻的剩磁值 | 2.444, 2.1788, ... |
| 温度(℃) | 测量时的温度 | 9.0, 7.5, ... |
| 相对湿度(%) | 测量时的相对湿度 | 90, 85, ... |

### 附件2：weather_data.xlsx

可选，如果提供应包含以下列：

| 列名 | 说明 |
|------|------|
| 天数 | 从0开始的天数 |
| 温度 | 该天的温度 |
| 相对湿度 | 该天的相对湿度百分比 |

**注**：如果不提供weather_data.xlsx，程序将使用附件1中的温度和湿度数据。

## 模型公式

### 最终预测模型（Model 5）

$$Y = \alpha_s \cdot t + \beta_{\log} \ln(1+t) + \beta_T C_T(t) + \beta_H TOW(t) + \beta_{TH} C_{TH}(t) + \beta_{rust} \, rust\_TOW(t) + \varepsilon$$

其中：

- $R_{i,s}(t) = M_{i,s}(t) / M_{i,s}(0)$：剩磁保持率
- $Y_{i,s}(t) = -\ln R_{i,s}(t)$：衰减强度
- $\alpha_s$：样品类型基础时间衰减斜率（通过 $0 + C(\text{样品类型}):\text{测量天数}$ 交互项估计）
- $C_T(t) = \sum_{d=1}^{t} z_T(d)$：累计温度暴露
- $TOW(t) = \sum_{d=1}^{t} I_{wet}(d)$：累计湿润时间，$I_{wet} = 1$ 当 $RH > 80\% \& T > 0$
- $C_{TH}(t) = \sum_{d=1}^{t} z_T(d) \times z_H(d)$：温湿度耦合项
- $rust\_TOW(t) = rust_0 \times TOW(t)$：锈蚀-湿润交互修正项
- $z_T, z_H$：标准化的温度和湿度
- $\ln(1+t)$：磁黏滞对数时间项

**模型角色**：
- Model 4：稳健基准模型（不含 $\ln(1+t)$），仅作对照
- Model 5：**最终预测模型**，加入统一 $\ln(1+t)$
- Model 6：复杂对照模型（type独立 $\ln(1+t)$），仅作对照

### 剩磁预测公式

$$\hat{M}_{i,s}(t) = M_{i,s}(0) \times \exp[-\hat{Y}_{i,s}(t)]$$

### 动态阈值修正

$$T_{dyn,s}(t) = T_{0,s} \times \exp[-\hat{Y}_s(t)]$$

其中 $T_{0,s}$ 为样品类型的静态阈值：
- 小号铁钉、小号铁夹：$T_0 = 1.0$ mT
- 普通钢筋、锈蚀钢筋：$T_0 = 1.5$ mT

## 修改字段映射

如果数据的列名与程序预期不同，可以在 `src/data_loader.py` 中的 `FieldMapper` 类中修改列名识别规则。

例如，如果实验数据中"样品类型"的列名为"Specimen_Type"：

```python
# 在 FieldMapper.__init__() 中添加
self.sample_type_names = ['样品类型', 'type', '类型', 'sample_type', 'Specimen_Type']
```

## 常见问题

### Q1: 程序找不到数据文件怎么办？

**A**: 确保 `附件1_模拟实验数据.csv` 和 `附件2-weather_data.xlsx` 在以下位置之一：
- 项目根目录
- `data/` 子目录

### Q2: 编码错误怎么办？

**A**: 程序自动检测文件编码（UTF-8、GBK等）。如果仍有问题，请确保CSV文件使用UTF-8编码保存。

### Q3: 某个模型构建失败怎么办？

**A**: 程序具有容错机制，如果某个模型失败（如MixedLM不收敛），会自动跳过。您可以在 `outputs/model_comparison.xlsx` 中查看哪些模型成功构建。

### Q4: 如何使用论文可以引用的格式输出结果？

**A**: 查看 `outputs/modeling_summary.md`，其中包含：
- 完整的模型公式（LaTeX格式）
- 模型对比表（Markdown格式）
- 参数解释和结论

可直接复制到论文中。

### Q5: 预测结果包含NaN怎么办？

**A**: 这通常表示某些天的天气数据缺失。程序会自动使用线性插值填补缺失值。如果仍有NaN，请检查原始天气数据的完整性。

## Python版本要求

- Python 3.7+
- 推荐 Python 3.8 或更高版本

## 依赖包说明

| 包名 | 用途 | 版本 |
|------|------|------|
| pandas | 数据处理 | ≥1.3.0 |
| numpy | 数值计算 | ≥1.21.0 |
| matplotlib | 数据可视化 | ≥3.4.0 |
| scikit-learn | 机器学习评估 | ≥0.24.0 |
| statsmodels | 统计建模 | ≥0.13.0 |
| openpyxl | Excel处理 | ≥3.6.0 |
| scipy | 科学计算 | ≥1.7.0 |
| chardet | 编码检测 | ≥4.0.0 |

## 输出文件详细说明

### 1. model_comparison.xlsx

| 列 | 含义 |
|----|------|
| Model | 模型编号 (Model 1-5) |
| R2 | 决定系数 |
| RMSE_Y | 在Y空间的均方根误差 |
| MAE_Y | 在Y空间的平均绝对误差 |
| RMSE_M | 在M空间的均方根误差 |
| MAE_M | 在M空间的平均绝对误差 |
| AIC | 赤池信息准则 |
| BIC | 贝叶斯信息准则 |

### 2. main_model_coefficients.xlsx

| 列 | 含义 |
|----|------|
| Coefficient | 参数名称 |
| Estimate | 参数估计值 |
| Std Error | 标准误差 |
| t-value | t统计量 |
| p-value | p值 |
| CI_Lower | 95%置信区间下界 |
| CI_Upper | 95%置信区间上界 |

### 3. prediction_23_29.xlsx

| 列 | 含义 |
|----|------|
| day | 预测的第几天 (23-29) |
| 小号铁钉 | 该天小号铁钉的预测剩磁值 (mT) |
| 小号铁夹 | 该天小号铁夹的预测剩磁值 (mT) |
| 普通钢筋 | 该天普通钢筋的预测剩磁值 (mT) |
| 锈蚀钢筋 | 该天锈蚀钢筋的预测剩磁值 (mT) |

### 4. dynamic_threshold_1_90.xlsx

| 列 | 含义 |
|----|------|
| day | 天数 (1-90) |
| 小号铁钉_阈值 | 该天小号铁钉的动态阈值 (mT) |
| 小号铁夹_阈值 | 该天小号铁夹的动态阈值 (mT) |
| 普通钢筋_阈值 | 该天普通钢筋的动态阈值 (mT) |
| 锈蚀钢筋_阈值 | 该天锈蚀钢筋的动态阈值 (mT) |

### 5. processed_data.csv

包含所有特征工程后的数据，便于后续分析。

## 高级用法

### 只运行特定阶段

在 `main.py` 中注释掉不需要的步骤：

```python
# 例如，只进行数据加载和预处理
df_exp, df_weather, col_maps = load_all_data()
df_exp_clean = preprocess_experiment_data(df_exp)
df_exp_clean = compute_remanence_ratio_and_decay(df_exp_clean)
df_merged = merge_weather_data(df_exp_clean, df_weather)
```

### 自定义模型

在 `src/modeling.py` 中的 `ModelBuilder` 类添加新模型：

```python
def build_model_6_custom(self):
    # 自定义模型公式
    df_model = self.df.copy()
    formula = 'Y ~ your_formula_here'
    model = ols(formula, data=df_model).fit()
    self.models['Model 6'] = model
    # ...
```

## 论文写作指南

### 推荐引用以下输出

1. **方法论**: `outputs/modeling_summary.md` 中的"4. 模型公式"部分
2. **结果表格**: 直接使用Excel文件中的表格或转换为LaTeX表格
3. **图表**: 直接使用 `outputs/figures/` 中的高质量PNG图表
4. **预测结果**: 表格形式呈现 `prediction_23_29.xlsx` 的数据
5. **动态阈值**: 用 `08_dynamic_thresholds.png` 说明阈值随时间的变化

### LaTeX 代码示例

```latex
% 引用最终预测模型公式 (Model 5)
\begin{equation}
Y_{i,s}(t) = \alpha_s \cdot t + \beta_{\log} \ln(1+t) + \beta_T C_T(t) + \beta_H \text{TOW}(t) + \beta_{TH} C_{TH}(t) + \beta_{rust} \, rust\_TOW(t) + \varepsilon
\end{equation}

% 插入图表
\begin{figure}
  \centering
  \includegraphics[width=0.8\textwidth]{figures/08_dynamic_thresholds.png}
  \caption{动态阈值变化}
\end{figure}
```

## 许可证

本项目用于数学建模竞赛，可自由修改和使用。

## 联系方式

如有问题或建议，请检查代码中的注释或查阅 `outputs/modeling_summary.md`。

---

**最后更新**: 2024年

**版本**: 1.0.0
