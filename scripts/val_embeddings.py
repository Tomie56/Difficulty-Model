import pandas as pd
import numpy as np

# Step 1: 加载数据
print("Step 1: 加载数据...")
df = pd.read_csv('../datas/train_lmsys_chat_with_rewards_cleaned.csv')
print(f"数据加载完成，包含 {len(df)} 条记录。")

# 仅使用需要的列
df = df[['conversation', 'reward']]

# 检查reward是否为空
df = df.dropna(subset=['reward'])
print(f"去除空值后，数据中剩余 {len(df)} 条记录。")

# Step 2: 加载文本的隐藏状态 (embeddings)
print("Step 2: 加载文本的隐藏状态...")
embeddings = np.load('../datas/embeddings.npy')
print(f"加载的 embeddings 形状: {embeddings.shape}")

# Step 3: 通过索引对比每个文本和对应的隐藏状态
print("Step 3: 验证文本和对应的嵌入是否一一对应...")

# 打印前 5 条文本及其对应的嵌入
for i in range(5):
    print(f"Text {i+1}: {df.iloc[i]['conversation']}")
    print(f"Embedding {i+1}: {embeddings[i]}")
    print("-" * 50)

# Step 4: 检查长度是否一致
print("Step 4: 检查长度是否一致...")
if len(df) == len(embeddings):
    print(f"数据集的长度 ({len(df)}) 与 embeddings 的长度 ({len(embeddings)}) 一致。")
else:
    print("数据集的长度与 embeddings 的长度不一致，请检查数据处理流程。")
