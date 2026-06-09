# 项目状态

更新日期：2026-06-09

---

## 版本演进

| 版本 | 关键改进 |
|------|---------|
| V2 基础版 | LightGBM `objective=regression` + 增量学习（init_model）+ MLflow 全程记录 |
| V3 量化版 | `objective` 改为 `quantile`，预测中位数更鲁棒；predict 脚本从 MLflow 加载 Production 模型做推理，训推分离 |

当前代码：`objective='regression'`（V2 状态，V3 量化版尚未合入）

---

## 当前能力

- [x] 多城市月度 Excel 加载与清洗（按城市分组均值填充）
- [x] KMeans 城市聚类（K=3，特征：教师人数/得教率/ROI 等）
- [x] 时序特征工程（lag_1/2/3, rolling_3_mean, rolling_6_std）
- [x] 交叉特征（新生老生增幅差异、VIP收入×报名增幅）
- [x] LightGBM TimeSeriesSplit 3折交叉验证训练
- [x] 增量学习（新月数据 → init_model 继续训练 100 轮）
- [x] Plotly 交互式预测对比图（实线=实际，虚线=预测）
- [x] Streamlit UI：文件上传、城市多选、结果下载

---

## 待完成

- [ ] V3 量化版（`quantile` objective）合入主分支
- [ ] MLflow 集成：参数/指标记录 + 模型注册 + Production 加载
- [ ] SHAP 可解释性分析（特征重要性图）
- [ ] 增量学习日期参数化（当前硬编码 `2025-10-01`）
- [ ] 模块化拆分（按 README.md 设计：preprocessing/models/pipeline/plotting）
- [ ] 跨平台路径处理（去除 Windows 硬编码路径）
- [ ] KMeans K 值对齐（代码 K=3，设计文档描述 K=4）

---

## 技术架构图

```
原始数据（多城市月度 Excel）
        │
        ▼
┌───────────────────────────────┐
│  load_data()                  │
│  · pd.read_excel              │
│  · 时间列 → datetime          │
│  · 缺失值 → 分城市均值填充     │
│  · 城市/LEVEL/节假日 → category│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  cluster_cities()  KMeans K=3 │
│  特征：教师人数/得教率/ROI/    │
│        渠道占比/新老生增幅/VIP │
│  → 每城市打上 cluster 标签    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  create_features()            │
│  时序：lag_1/2/3              │
│        rolling_3_mean         │
│        rolling_6_std          │
│  交叉：新生老生增幅差异        │
│        VIP收入×报名增幅        │
│  时间：month / quarter        │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  train_lgbm()                 │
│  · TimeSeriesSplit  n=3       │
│  · objective = regression     │  ← V3 改为 quantile
│  · early_stopping = 50        │
│  → 返回 3 个 fold 模型列表    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  update_models_with_latest_data()  增量学习         │
│  · init_model = 已有 booster  │
│  · num_boost_round = 100      │
│  · 新月数据追加训练            │
└───────────────┬───────────────┘
                │
           ┌────┴────┐
           │         │
           ▼         ▼
  predict_latest   forecast_cities()
  _data()          · 滚动生成未来特征
  · 当月回测        · 3模型均值集成预测
           │         │
           └────┬────┘
                │
                ▼
┌───────────────────────────────┐
│  MLflow（设计目标，待实现）    │
│  · 记录参数/指标               │
│  · 注册模型至 Model Registry  │
│  · Production 模型加载推理    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  plot_forecasts_interactive() │
│  · Plotly：实线=实际           │
│            虚线=预测           │
│  · 同城市同色，hover 统一      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Streamlit UI                 │
│  · 文件上传（历史 + 当月）     │
│  · 城市多选 multiselect       │
│  · plotly_chart               │
│  · download_button → xlsx     │
└───────────────────────────────┘

          （SHAP 可解释性 — 待实现）
```
