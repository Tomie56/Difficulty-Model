import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: 加载数据
print("Step 1: 加载数据...")
df = pd.read_csv('../datas/test_lmsys_chat_with_rewards_cleaned.csv')
print(f"数据加载完成，包含 {len(df)} 条记录。")

# 仅使用需要的列
df = df[['conversation', 'reward']]

# 检查reward是否为空
df = df.dropna(subset=['reward'])
print(f"去除空值后，数据中剩余 {len(df)} 条记录。")

# Step 2: 加载模型和tokenizer
print("Step 2: 加载模型和tokenizer...")
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"模型 {model_name} 加载完成。")

# 检查并设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("没有pad_token，使用eos_token作为pad_token。")
else:
    print("pad_token已设置。")

# 将模型移到GPU，如果有的话
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"模型已移到 {device}。")

# Step 3: 定义函数，获取文本的隐藏状态
print("Step 3: 定义函数，获取文本的隐藏状态...")

def get_hidden_states(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    last_hidden_state = outputs.hidden_states[-1]  # 最后一层的隐藏状态
    sentence_embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # 获取句子级的嵌入
    return sentence_embedding

# Step 4: 获取所有文本的隐藏状态并保存
embeddings = []

# 使用 tqdm 包装 DataFrame 的 iterrows() 方法来显示进度条
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Texts"):
    text = row['conversation']
    embedding = get_hidden_states(text)
    embeddings.append(embedding)

# 保存隐藏状态到文件
embeddings = np.array(embeddings)
np.save('../datas/embeddings_test.npy', embeddings)
print("文本的隐藏状态已保存到 '../datas/train_embeddings.npy' 文件")
