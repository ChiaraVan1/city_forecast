# 导入核心库 
import pandas as pd 
import numpy as np 
import lightgbm as lgb 
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
import warnings 
import plotly.graph_objects as go 
from plotly.graph_objects import Figure, Scatter
import os
import webbrowser
import dash
from dash import dcc, html
from dash.dependencies import Input, Output 
import plotly.express as px 
import plotly.offline as pyoff 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score 
import streamlit as st


# 配置环境 
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

# 全局定义模型参数 
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity': -1 
}

# 数据加载与预处理 
def load_data(file_path):
    df = pd.read_excel(file_path)
    
    # 转换日期格式 
    df["时间"] = pd.to_datetime(df["时间"].astype(str), format="%Y%m")
    df = df.sort_values(['城市', '时间'])
    
    # 缺失值处理（分城市填充）
    numeric_cols = ['月均教师人数', '得教率', '主动报入渠道占比', '新生人头增幅', '老生人头增幅', '市场费ROI', 'VIP收入占比', '报名收入增幅']
    for col in numeric_cols:
        df[col] = df.groupby('城市')[col].transform(lambda x: x.fillna(x.mean()))
    
    # 城市类别编码 
    df['城市'] = df['城市'].astype('category')
    df['LEVEL'] = df['LEVEL'].astype('category')
    df['is_holiday_SEASON'] = df['is_holiday_SEASON'].astype('category')   
   
    return df 

# 标准化
def extract_and_scale_features(df):
    # 提取特征 
    cluster_features = df.groupby("城市").agg({
        '月均教师人数': 'mean',
        '得教率': 'mean',
        '主动报入渠道占比': 'mean',
        '新生人头增幅': 'mean',
        '老生人头增幅': 'mean',
        '市场费ROI': 'mean',
        'VIP收入占比': 'mean'
    }).reset_index()
    # 特征标准化 
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features.drop(columns=['城市']))
    return cluster_features, scaled_features 

# 进行聚类
def cluster_cities(df, n_clusters=3):
    # 提取特征并进行标准化 
    cluster_features, scaled_features = extract_and_scale_features(df)
    # 使用 KMeans 进行聚类 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_features['cluster'] = kmeans.fit_predict(scaled_features)
    # 将聚类结果合并回原始数据 
    df = df.merge(cluster_features[['城市', 'cluster']], on='城市', how='left')
    return df, cluster_features, scaled_features 

# 在create_features函数中添加调试信息
def create_features(df, target_col='报名收入增幅', lag_periods=3):  
    # 确保 cluster 列被保留 
    if 'cluster' not in df.columns:
        raise ValueError("df 中缺少 cluster 列")
    # 按时间排序 
    df = df.sort_values(by='时间')
    # 滞后特征 
    for i in range(1, lag_periods + 1):
        col = f'lag_{i}'
        df[col] = df.groupby('城市')[target_col].shift(i)
        df[col] = df[col].ffill().bfill()
    # 滚动统计量 
    df['rolling_3_mean'] = df.groupby('城市')[target_col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['rolling_3_mean'] = df['rolling_3_mean'].ffill().bfill()
    df['rolling_6_std'] = df.groupby('城市')[target_col].transform(lambda x: x.rolling(6, min_periods=1).std())
    df['rolling_6_std'] = df['rolling_6_std'].ffill().bfill()
    # 时间特征 
    df['month'] = df['时间'].dt.month.astype('category')
    df['quarter'] = df['时间'].dt.quarter.astype('category')
    # 新增特征 
    df['教师人数_得教率'] = df['月均教师人数'] * df['得教率'] 
    df['新生老生增幅差异'] = df['新生人头增幅'] - df['老生人头增幅']
    df['VIP收入_报名增幅'] = df['VIP收入占比'] * df['报名收入增幅']
    # 合并主动报入渠道占比和得教率为综合指标 
    df['综合教学效率'] = df['主动报入渠道占比'] * df['得教率']
    # 添加聚类结果作为特征 
    if 'cluster' in df.columns:
        df['cluster'] = df['cluster'].astype('category')
    df['LEVEL'] = df['LEVEL'].astype('category')
    df['is_holiday_SEASON'] = df['is_holiday_SEASON'].astype('category')
    return df

# 模型训练与验证 
def train_lgbm(df, target_col='报名收入增幅'):   
    # 确保 cluster 列被保留
    if 'cluster' not in df.columns:
        raise ValueError("df_eng 中缺少 cluster 列")
    # 特征列定义 
    features = ['月均教师人数', '得教率', '主动报入渠道占比', '新生人头增幅', '老生人头增幅', 
                '市场费ROI', 'VIP收入占比', 'month', 'quarter', 
                'lag_1', 'lag_2', 'lag_3', 
                'rolling_3_mean', 'rolling_6_std', 
                # '教师人数_得教率', 
                '新生老生增幅差异', 'VIP收入_报名增幅', '城市', 'cluster', 'LEVEL', 'is_holiday_SEASON']
    
    X = df[features]
    y = df[target_col]
    
    # 时间序列交叉验证 
    tscv = TimeSeriesSplit(n_splits=3)
    metrics = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # LightGBM 数据集 
        train_set = lgb.Dataset(
            X_train, y_train,
            categorical_feature=['城市', 'month', 'quarter', 'cluster', 'LEVEL', 'is_holiday_SEASON']
        )       
        
        # 训练 
        model = lgb.train(
            params,
            train_set,
            valid_sets=[train_set],
            callbacks=[lgb.early_stopping(50)],
            num_boost_round=1000 
        )
        
        # 验证集预测 
        val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        metrics.append({'fold': fold+1, 'MSE': mse, 'R2': r2})
        models.append(model)
    
    return models, pd.DataFrame(metrics)

def forecast_cities(models, df, forecast_months=1):
    # 获取最后日期并生成未来时间点 
    last_date = df['时间'].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq='MS'
    )
    # 打印future_dates 
    print("生成的未来日期：", future_dates)    
    # 存储预测结果 
    all_forecasts = []    
    for city in df['城市'].cat.categories:
        print(f"\n开始处理城市: {city}")       
        # 构建未来数据 
        city_data = df[df['城市'] == city].copy()
        if city_data['cluster'].isnull().any():
            print(f"城市 {city} 的 cluster 值存在空值")
        city_data['时间'] = pd.to_datetime(city_data['时间'], format='%Y-%m')
        city_data = city_data.sort_values(by='时间')
        # 取最近3期数据生成滞后特征 
        city_recent_data = city_data.iloc[-3:]   
        # 确保 cluster 列存在
        if 'cluster' not in city_recent_data.columns:
            raise ValueError(f"城市 {city} 的数据中缺少 cluster 列")    
        # 获取当前城市的 cluster 值
        current_cluster = city_recent_data['cluster'].iloc[0]
        current_LEVEL = city_recent_data['LEVEL'].iloc[0]
        future_df = pd.DataFrame({
            '时间': future_dates,
            '城市': city,
            'cluster': current_cluster,
            'LEVEL': current_LEVEL,
            '月均教师人数': city_recent_data['月均教师人数'].mean(),
            '得教率': city_recent_data['得教率'].mean(),
            '主动报入渠道占比': city_recent_data['主动报入渠道占比'].mean(),
            '新生人头增幅': city_recent_data['新生人头增幅'].mean(),
            '老生人头增幅': city_recent_data['老生人头增幅'].mean(),
            '市场费ROI': city_recent_data['市场费ROI'].mean(),
            'VIP收入占比': city_recent_data['VIP收入占比'].mean()
        })                    
        # 合并数据
        full_df = pd.concat([df, future_df], ignore_index=True)
        full_df = full_df.sort_values(by='时间')    
        full_df['城市'] = full_df['城市'].astype('category')   
        full_df['cluster'] = full_df['cluster'].astype('category')  
        full_df['LEVEL'] = full_df['LEVEL'].astype('category')  
        full_df['is_holiday_SEASON'] = full_df['is_holiday_SEASON'].astype('category')  
        # 创建特征 
        full_df = create_features(full_df)       
        # 提取预测所需特征 
        future_features = full_df[full_df['时间'].isin(future_dates)]       
        X_future = future_features[models[0].feature_name()]        
        if X_future.empty:
            print(f"警告：城市 {city} 的预测数据为空，跳过预测。")
            continue         
        # 集成模型预测 
        preds = np.mean([model.predict(X_future) for model in models], axis=0)       
        # 保存结果 
        city_forecast = pd.DataFrame({
            '时间': future_dates,
            '城市': city,
            '预测值': preds 
        })
        all_forecasts.append(city_forecast)
        print(f"成功存储 {city} 的预测结果。")    
    if not all_forecasts:
        print("警告：没有任何城市的预测结果.")
        return pd.DataFrame()       
    return pd.concat(all_forecasts)


# 传入城市
def plot_forecasts_interactive(forecast_df, selected_cities):
    """
    绘制交互式预测图，区分实际值和预测值。
    - forecast_df: 已经过滤只包含 selected_cities 的 DataFrame
    - selected_cities: 保证颜色一致性
    """
    fig = go.Figure()

    # 用 px 分配颜色
    color_map = px.colors.qualitative.Set2
    city_color_dict = {
        city: color_map[i % len(color_map)]
        for i, city in enumerate(selected_cities)
    }

    for city in selected_cities:
        city_data = forecast_df[forecast_df['城市'] == city]

        # 实际数据：实线
        actual = city_data[city_data['来源'] == '实际']
        fig.add_trace(go.Scatter(
            x=actual['时间'],
            y=actual['报名收入增幅'],
            mode='lines+markers',
            name=f'{city} - 实际',
            line=dict(color=city_color_dict[city], dash='solid'),
            marker=dict(symbol='circle', size=6)
        ))

        # 预测数据：虚线
        pred = city_data[city_data['来源'] == '预测']
        fig.add_trace(go.Scatter(
            x=pred['时间'],
            y=pred['报名收入增幅'],
            mode='lines+markers',
            name=f'{city} - 预测',
            line=dict(color=city_color_dict[city], dash='dash'),
            marker=dict(symbol='diamond', size=6)
        ))

    fig.update_layout(
        title='城市报名收入增幅预测 vs 实际',
        xaxis_title='时间',
        yaxis_title='报名收入增幅',
        legend_title='城市与数据类型',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )


    return fig

def update_models_with_latest_data(models, df):
    """
    使用实际数据（10 月）更新模型。
    """
    # 获取最新数据 
    latest_data = df[df['时间'] == pd.Timestamp('2025-10-01')]
    if latest_data.empty:
        print("警告：未找到 10 月的数据，无法更新模型。")
        return models 
    
    print("开始更新模型...")
    for i, model in enumerate(models):
        # 提取特征和目标值 
        X_new = latest_data[model.feature_name()]
        y_new = latest_data['报名收入增幅']
        
        # 更新模型 
        model = lgb.train(
            params,
            train_set=lgb.Dataset(X_new, y_new),
            init_model=model,
            keep_training_booster=True,
            num_boost_round=100  # 可以调整更新的轮数 
        )
        models[i] = model  # 更新模型列表中的模型 
        print(f"模型 {i} 更新完成。")
    
    return models 

def predict_latest_data(models, df):
    """
    使用更新后的模型预测 10 月的数据。
    """
    # 获取最新数据 
    latest_data = df[df['时间'] == pd.Timestamp('2025-10-01')]

    # 提取特征 
    X_latest = latest_data[models[0].feature_name()]
    
    # 使用更新后的模型进行预测 
    preds = np.mean([model.predict(X_latest) for model in models], axis=0)
    
    # 保存结果 
    predictions = pd.DataFrame({
        '时间': latest_data['时间'],
        '城市': latest_data['城市'],
        '实际值': latest_data['报名收入增幅'],
        '预测值': preds 
    })
    
    return predictions 


def run_pipeline(history_file, oct_file):
    # 加载 + 合并 + 预处理
    df = load_data(history_file)
    df = df[df['时间'].dt.year != 2023]
    df = df[df['时间'] != pd.Timestamp('2025-10-01')]

    new_data = load_data(oct_file)
    df = pd.concat([df, new_data], ignore_index=True)

    df, cluster_features, _ = cluster_cities(df)
    df_eng = create_features(df)

    # 训练 + 更新 + 预测
    models, _ = train_lgbm(df_eng)
    models = update_models_with_latest_data(models, df_eng)
    predictions = predict_latest_data(models, df_eng)

    # 新增标签
    actual = df[['时间', '城市', '报名收入增幅']].copy()
    actual['来源'] = '实际' 
    pred = predictions.rename(columns={'预测值': '报名收入增幅'}).copy()
    pred['来源'] = '预测'


    # 合并预测 + 原始数据
    merged_df = pd.concat([actual, pred], ignore_index=True)

    # 保存为 Excel
    export_path = r"d:/工作/WPScloud/1622414952/WPS企业云盘/新东方教育科技集团有限公司/我的企业文档/2025/04/收入增幅/M10_updated_predictions.xlsx"
    merged_df.to_excel(export_path, sheet_name="预测与历史对比", index=False)
    
    # 获取可选择的城市列表
    city_list = merged_df['城市'].unique().tolist()
    
    # 调用 plot_forecasts_interactive 函数并返回 HTML 字符串
    plot_html = plot_forecasts_interactive(merged_df, city_list)
    
    # 返回需要的内容
    return merged_df, city_list, export_path

# 初始化 session_state
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None
if 'city_list' not in st.session_state:
    st.session_state.city_list = []
if 'export_path' not in st.session_state:
    st.session_state.export_path = None

st.set_page_config(page_title = '城市报名收入预测', layout = 'wide')
st.title("📈 城市报名收入监控平台")

# 文件上传
uploaded_file = st.file_uploader("上传历史数据 Excel", type=["xlsx"])
uploaded_oct_file = st.file_uploader("上传当月（10月）数据 Excel", type=["xlsx"])

# 设置默认值
default_file = r"D:\工作\WPScloud\1622414952\WPS企业云盘\新东方教育科技集团有限公司\我的企业文档\2025\04\收入增幅\Mdata1 2025-03-25 15_51_15.xlsx"
default_oct = r"D:\工作\WPScloud\1622414952\WPS企业云盘\新东方教育科技集团有限公司\我的企业文档\2025\04\收入增幅\Mdata1 2025-10.xlsx"

# 开始按钮
if st.button("🚀 运行预测分析"):
    with st.spinner("模型运行中，请稍候..."):
        if not uploaded_file or not uploaded_oct_file:
            st.warning("请上传历史数据和10月数据。使用默认数据进行演示。")
        history_path = uploaded_file if uploaded_file else default_file
        oct_path = uploaded_oct_file if uploaded_oct_file else default_oct

        # 运行分析管道
        try:
            merged_df, city_list, export_path = run_pipeline(history_path, oct_path)
            
            # 保存结果到 session_state
            st.session_state.merged_df = merged_df
            st.session_state.city_list = city_list
            st.session_state.export_path = export_path

        except Exception as e:
            st.error(f"运行过程中发生错误：{e}")

# 如果 session_state 中有数据，则显示图表
if st.session_state.merged_df is not None:
    selected_cities = st.multiselect(
        "选择要展示的城市",
        st.session_state.city_list,
        default=st.session_state.city_list[:3]
    )
    
    if selected_cities:
        # 生成 plotly 图表
        fig = plot_forecasts_interactive(st.session_state.merged_df, selected_cities)
        st.plotly_chart(fig, use_container_width=True)

    # 下载按钮
    with open(st.session_state.export_path, "rb") as f:
        st.download_button("📥 下载完整预测数据", f, file_name="城市预测结果.xlsx")


