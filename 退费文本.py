'''conda create -n bertopic_env python=3.9
conda activate bertopic_env
pip install bertopic
# 强制调用解释器 D:\anaconda\envs\bertopic_env\python r"D:\code\python\退费文本.py"
'''

# -*- coding: utf-8 -*-
"""
这是一个完整的混合策略脚本，用于处理包含“内部日志”和“用户原因”的复杂文本数据。

最终目标：为每一条原始文本数据，打上一个清晰、准确的分类标签。

核心策略：分而治之 (Divide and Conquer)
1.  **数据分流 (Triage)**：用规则识别出文本是“日志”还是“潜在原因”。
2.  **并行处理 (Parallel Processing)**：
    a. **日志流**：用更详细的规则进行结构化解析，提取操作、部门等信息。
    b. **原因流**：先用规则“萃取”出被包裹的核心原因，再用AI模型（此处为模拟的BERTopic）进行语义聚类。
3.  **结果整合 (Synthesis)**：将两路处理的结果合并，形成最终的统一分类。
"""

# ==============================================================================
# 步骤 0: 导入所有需要的库
# ==============================================================================
import re
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np

# ==============================================================================
# 步骤 1: 准备数据
# ==============================================================================
print("--- 步骤 1: 加载模拟数据 ---")
df = pd.read_excel(r"D:\工作\WPScloud\1622414952\WPS企业云盘\我的企业文档\2025\07\退费\biz_memo.xlsx")
print("原始数据加载完毕：")
print(df)
print("\n" + "="*80 + "\n")

# ==============================================================================
# 步骤 2: 定义所有我们将要用到的“工具”函数
# ==============================================================================
print("--- 步骤 2: 定义所有处理函数 ---")

# --- 工具 2a: 数据分拣机 (V3 - 更严格的版本) ---
def classify_text_type(text):
    """根据规则判断文本是'日志'还是'潜在原因'。"""
    text = str(text).strip()
    if not text: return "空值"
    
    # 规则1: 包含明确的日志结构符号
    if '/' in text or re.search(r'^\d+$', text): return "日志"
    
    # 规则2: 包含明确的日志关键词
    log_keywords = ['续费', '新签', '储值', '顾问', '推荐', '支付', '扫码', '后台', '老师', '转班', '退班', '市场活动', '校代', '提现', '重报', '返款']
    if any(keyword in text for keyword in log_keywords): return "日志"
    
    # 规则3: 格式像 '部门-姓名' 或被符号包裹
    if re.fullmatch(r'[\u4e00-\u9fa5]+-[\u4e00-\u9fa5\s\d]+', text) or re.fullmatch(r'《.+》', text):
        return "日志"
        
    # 规则4: 包含多个逗号或连字符
    if text.count(',') >= 2 or text.count('-') >= 2 or re.search(r'NO\.\d+', text): return "日志"
    
    # 规则5: 数字和字母占比较高
    alnum_count = len(re.findall(r'[a-zA-Z0-9]', text))
    if len(text) > 0 and (alnum_count / len(text)) > 0.5: return "日志"
        
    return "潜在原因"

# --- 工具 2b: 原因萃取器 (V3 - 更智能的版本) ---
def extract_pure_reason(text):
    """从可能被日志外壳包裹的文本中，萃取出核心原因。"""
    text = str(text).strip()
    
    # 规则 1: 匹配 "因..." 或 "由于..."
    match = re.search(r'(?:因|由于)(.+?)(?:问题|原因|,|，|$)', text)
    if match: return match.group(1).strip('* ')

    # 规则 2: 匹配特定的原因关键词
    reason_keywords = ['时间冲突', '时间不合适', '更换时间', '换时间上课', '课程取消', '效果不好', '内容不符', '讲得太快', '跟不上', '从头开始上']
    for keyword in reason_keywords:
        if keyword in text: return keyword
            
    # 规则 3: 匹配特殊符号包裹的内容
    match = re.search(r'【(.+?)】|——(.+)', text)
    if match: return (match.group(1) or match.group(2)).strip()

    # 规则 4: 处理用 '+' 连接的日志
    if '+' in text:
        last_part = text.rsplit('+', 1)[-1]
        if not re.fullmatch(r'[a-zA-Z0-9\s]*', last_part):
            last_part_cleaned = re.sub(r'\(.*\)$', '', last_part).strip()
            if last_part_cleaned: return last_part_cleaned

    # 规则 5: 清理常见的日志干扰
    text_no_name = re.sub(r'^[^\s-]+-|\s*——\s*[^\s]+$', '', text).strip()
    if len(text_no_name.split()) == 1 and len(text_no_name) < 5:
        return text 
    return text_no_name if len(text_no_name) > 2 else text

# --- 工具 2c: 日志解析规则字典 ---
BUSINESS_RULES = {
    'action': {
        '退费': ['退费', '退班', '退回'], '转班': ['转班', '换时间', '更换时间', '时间调整', '时间冲突', '时间不合适'],
        '续费': ['续费'], '新签': ['新签', '新生', '报班'], '重报': ['重报', '退出重报']
    },
    'department': {'国际教育': ['国际教育'], '青少': ['青少'], '国外部': ['国外部', '国外考试部']},
    'channel': {'市场活动': ['市场活动'], '内部推荐': ['内部推荐'], '校代': ['校代']}
}

# --- 工具 2d: 日志解析器 ---
def parse_log_details(log_text):
    log_text = str(log_text)
    parsed_info = {'action': '未知', 'department': '未知', 'channel': '未知', 'amount': None}
    for dimension, categories in BUSINESS_RULES.items():
        for category, keywords in categories.items():
            if any(keyword in log_text for keyword in keywords):
                parsed_info[dimension] = category; break
    match = re.search(r'(-?[\d,]+\.?\d*)', log_text)
    if match:
        amount_str = match.group(1).replace(',', '')
        if amount_str: parsed_info['amount'] = float(amount_str)
    return parsed_info

# --- 工具 2e: 全面主题分析器 ---
def analyze_all_topics(topic_model, docs):
    """
    为所有主题生成一份详细的“体检报告”，包含技术化指标。
    """
    print("\n" + "#"*80)
    print("### 全面主题分析报告 ###")
    print("#"*80 + "\n")
    
    topic_freq = topic_model.get_topic_info()
    
    for index, row in topic_freq.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            print("--- 跳过主题 -1 (未分类文档) ---\n")
            continue
            
        print(f"--- 主题ID: {topic_id} | 文档数量: {row['Count']} ---")
        
        keywords = [f"{word}({score:.3f})" for word, score in topic_model.get_topic(topic_id)]
        print(f"  [关键词]: {', '.join(keywords[:10])}")
        
        print("  [代表性文档]:")
        representative_docs = topic_model.get_representative_docs(topic_id)
        for doc in representative_docs:
            print(f"    - {doc}")
            
        print("  [技术化指标]:")
        topic_docs = [docs[i] for i, t in enumerate(topic_model.topics_) if t == topic_id]
        if topic_docs:
            avg_len = np.mean([len(str(d)) for d in topic_docs])
            print(f"    - 平均文本长度: {avg_len:.2f}")
            avg_digit_ratio = np.mean([len(re.findall(r'\d', str(d))) / len(str(d)) if len(str(d)) > 0 else 0 for d in topic_docs])
            print(f"    - 平均数字占比: {avg_digit_ratio:.2%}")
        
        print("-" * 60 + "\n")

print("所有函数定义完毕。\n" + "="*80 + "\n")


# ==============================================================================
# 步骤 3: 执行数据分流与并行处理
# ==============================================================================
print("--- 步骤 3: 执行混合策略工作流 ---")

# --- 3a. 数据分流 ---
print("\n[3a] 执行数据分流...")
df['data_type'] = df['refund_reason'].apply(classify_text_type)
print("分流结果统计:")
print(df['data_type'].value_counts())

df_logs = df[df['data_type'] == '日志'].copy()
df_reasons = df[df['data_type'] == '潜在原因'].copy()
print(f"\n分离出 {len(df_logs)} 条日志 和 {len(df_reasons)} 条潜在原因。")

# --- 3b. 处理“日志”流 ---
print("\n[3b] 处理日志流...")
if not df_logs.empty:
    log_details = df_logs['refund_reason'].apply(parse_log_details)
    parsed_df = pd.json_normalize(log_details)
    df_logs_classified = pd.concat([df_logs.reset_index(drop=True), parsed_df], axis=1)
    df_logs_classified['log_category'] = "操作:" + df_logs_classified['action'] + "|部门:" + df_logs_classified['department'] + "|渠道:" + df_logs_classified['channel']
    print("日志数据解析与分类完成。")
else:
    print("没有需要处理的日志。")

# --- 3c. 处理“原因”流 ---
print("\n[3c] 处理原因流...")
if not df_reasons.empty:
    df_reasons['pure_reason'] = df_reasons['refund_reason'].apply(extract_pure_reason)
    print("原因萃取完成。")
    
    print("\n开始运行BERTopic模型，这可能需要几分钟...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    topic_model = BERTopic(embedding_model=embedding_model, language="multilingual", min_topic_size=50, verbose=True)
    
    pure_reasons_docs = df_reasons['pure_reason'].dropna().tolist()
    topics, _ = topic_model.fit_transform(pure_reasons_docs)
    
    df_reasons['topic_id'] = topics
    print("BERTopic模型训练完成！")

    analyze_all_topics(topic_model, pure_reasons_docs)
    
    print("\n--- 请根据上面的分析报告，在代码中手动创建你的 topic_map ---")
    topic_map = {
        5: "原因:时间冲突", 13: "原因:时间冲突", 17: "原因:时间冲突", 31: "原因:时间冲突",
        3: "原因:学员计划变更", 4: "原因:学员计划变更", 10: "原因:学员计划变更", 38: "原因:学员计划变更",
        53: "原因:考试意向改变",
        6: "原因:更换班级/校区", 56: "原因:报名操作失误",
        43: "原因:课程质量/难度",
    }
    print("已定义的 topic_map:", topic_map)
    
    df_reasons['reason_category'] = df_reasons['topic_id'].map(topic_map).fillna("原因:其他")
    print("\n语义分类完成。")
else:
    print("没有需要处理的潜在原因。")

print("\n并行处理完成。\n" + "="*80 + "\n")


# ==============================================================================
# 步骤 4: 整合最终结果
# ==============================================================================
print("--- 步骤 4: 整合最终结果 ---")

# 先将主DataFrame与日志分类结果合并
if not df_logs.empty:
    log_cats = df_logs_classified[['refund_reason', 'log_category']].drop_duplicates(subset=['refund_reason'])
    df = pd.merge(df, log_cats, on='refund_reason', how='left')
else:
    df['log_category'] = np.nan

# 接着，将主DataFrame与原因分类结果合并
if not df_reasons.empty:
    reason_cats = df_reasons[['refund_reason', 'reason_category']].drop_duplicates(subset=['refund_reason'])
    df = pd.merge(df, reason_cats, on='refund_reason', how='left')
else:
    df['reason_category'] = np.nan

# 现在，整合两路结果到最终的分类列
df['final_category'] = df['log_category'].fillna(df['reason_category'])
df['final_category'].fillna('未分类', inplace=True)

# 清理临时的中间列
df.drop(columns=['log_category', 'reason_category'], errors='ignore', inplace=True)

print("所有结果已整合到主DataFrame。")


# ==============================================================================
# 步骤 5: 最终分析与总结
# ==============================================================================
print("\n--- 步骤 5: 最终分析与总结 ---")
print(df['final_category'].value_counts())
output_filename = r"D:\工作\WPScloud\1622414952\WPS企业云盘\我的企业文档\2025\07\退费\refund_reason_classification_result.xlsx"
df.to_excel(output_filename, index=False)
print(f"\n任务完成！详细分类结果已保存至文件: {output_filename}")
