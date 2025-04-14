import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.nn import MSELoss
import pandas as pd

# 1. 加载训练数据
df = pd.read_csv("../datas/train_lmsys_chat_with_rewards_cleaned.csv")
rewards = df["reward"].tolist()

# 2. 直接从embeddings.npy中读取隐藏状态
embeddings = np.load('../datas/embeddings.npy')  # 读取保存的嵌入文件
embeddings = torch.tensor(embeddings, dtype=torch.float32)  # 转换为torch张量

# 3. 设置LoRA配置
lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=16,  # 控制LoRA适配器的学习率
    target_modules=["q_proj", "v_proj"],  # 选择应用LoRA的模块
    lora_dropout=0.1,
    bias="none"  # 不使用偏置项
)

# 4. 加载预训练的GPT-Neo-2.7B模型
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

# 5. 将LoRA应用到模型中
model = get_peft_model(model, lora_config)

# 6. 创建数据集并构造DataLoader
labels = torch.tensor(rewards, dtype=torch.float32)  # 对应的reward值
train_dataset = TensorDataset(embeddings, labels)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 7. 设置优化器和损失函数
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = MSELoss()

# 8. 训练循环
model.train()
for epoch in range(3):  # 假设训练3个epoch
    for batch in train_dataloader:
        inputs, reward = batch
        optimizer.zero_grad()
        
        # 确保输入是LongTensor类型，适应Embedding层要求
        inputs = inputs.long()  # 将浮点型输入转换为 LongTensor
        
        # 使用GPT-Neo进行前向传播
        outputs = model(inputs)
        logits = outputs.logits[:, -1].squeeze(-1)  # 获取最后一个token的预测值
        
        # 计算损失
        loss = loss_fn(logits, reward)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 打印训练过程中的损失
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 9. 保存微调后的模型

torch.save(model.state_dict(), "lora_reward_predictor.pt")

