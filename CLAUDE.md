# city_forecast — Claude Code 项目文档

## 项目概述

新东方多城市报名收入增幅预测平台。从多城市月度 Excel 数据出发，经过清洗、城市聚类、特征工程，用 LightGBM 做时间序列预测，支持增量学习。前端用 Streamlit 展示实际 vs 预测对比图，支持文件上传与城市多选。

---

## 文件结构

| 文件 | 作用 |
|------|------|
| `city_forecast.py` | 主应用：数据管道 + LightGBM 训练 + Streamlit UI |
| `README.md` | 计划中的模块化目录结构（尚未拆分实现） |

---

## 运行方式

```bash
streamlit run city_forecast.py
```

---

## 数据结构（Excel 输入列）

| 列名 | 类型 | 说明 |
|------|------|------|
| `时间` | YYYYMM | 月份，转为 datetime |
| `城市` | category | 城市名 |
| `LEVEL` | category | 城市级别 |
| `is_holiday_SEASON` | category | 是否节假日季 |
| `月均教师人数` | float | 当月平均教师数 |
| `得教率` | float | 教学转化率 |
| `主动报入渠道占比` | float | 主动报名渠道占比 |
| `新生人头增幅` | float | 新生人数增幅 |
| `老生人头增幅` | float | 老生人数增幅 |
| `市场费ROI` | float | 市场营销 ROI |
| `VIP收入占比` | float | VIP 课程收入占比 |
| `报名收入增幅` | float | **目标变量** |

---

## 核心函数调用链

```
load_data()
    → cluster_cities()                          # KMeans K=3，城市聚类
        → create_features()                     # lag/rolling/交叉特征
            → train_lgbm()                      # TimeSeriesSplit 3折
                → update_models_with_latest_data()   # 增量学习 100轮
                    → predict_latest_data()          # 当月推理
                        → plot_forecasts_interactive()   # Plotly 对比图
```

---

## 模型参数（`params` 全局字典）

```python
objective     = 'regression'   # V3 量化版改为 'quantile'
metric        = 'rmse'
learning_rate = 0.05
num_leaves    = 31
feature_fraction = bagging_fraction = 0.8
```

---

## 特征列表

| 类别 | 特征 |
|------|------|
| 时序 | `lag_1`, `lag_2`, `lag_3`, `rolling_3_mean`, `rolling_6_std` |
| 原始 | `月均教师人数`, `得教率`, `主动报入渠道占比`, `新生人头增幅`, `老生人头增幅`, `市场费ROI`, `VIP收入占比` |
| 交叉 | `新生老生增幅差异`, `VIP收入_报名增幅` |
| 时间 | `month`, `quarter` |
| 类别 | `城市`, `cluster`, `LEVEL`, `is_holiday_SEASON` |

> `教师人数_得教率` 已注释掉（city_forecast.py:132）

---

## 增量学习机制

`update_models_with_latest_data()` 通过 `lgb.train(..., init_model=model, keep_training_booster=True, num_boost_round=100)` 追加训练。当前日期硬编码为 `2025-10-01`，需参数化。

---

## 已知问题 / 待做

- 增量学习日期硬编码（`2025-10-01`），需改为动态获取最新月份
- Windows 路径硬编码（`D:\工作\...`），本地路径需跨平台处理
- MLflow 集成（设计目标）尚未在代码中实现
- SHAP 可解释性分析尚未实现
- README.md 描述的模块化拆分尚未执行
