city_forecast_app/
│
├── app.py                       # 主 Streamlit 应用入口
├── config.py                    # 配置参数（如模型路径、全局参数等）
├── requirements.txt             # 所需依赖
│
├── data/                        # 存放原始和预处理后数据
│   ├── raw/
│   └── processed/
│
├── models/                      # 模型相关（训练/加载/预测）
│   ├── model.py
│   └── trainer.py
│
├── pipeline/                    # 数据流控制（加载、预测全流程）
│   └── pipeline.py
│
├── preprocessing/              # 特征工程与数据清洗
│   └── preprocessing.py
│
├── plotting/                   # 所有可视化函数（plotly 或 matplotlib）
│   └── plot.py
│
├── utils/                      # 工具函数（如日志、缓存、时间处理等）
│   └── utils.py
│
└── notebooks/                  # 实验/调试用 VBA

