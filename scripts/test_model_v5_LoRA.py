import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch.optim as optim

# Step 1: 加载测试集数据
df_test = pd.read_csv('../datas/test_lmsys_chat_with_rewards_cleaned.csv')
df_test = df_test[['conversation', 'reward']].dropna(subset=['reward'])
test_embeddings = np.load('../datas/test_embeddings.npy')  # 假设test_embeddings.npy包含测试集文本的隐藏状态

y_test = df_test['reward'].values

# Step 2: 数据集处理
test_data = TensorDataset(torch.tensor(test_embeddings, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
test_loader = DataLoader(test_data, batch_size=16)

# Step 3: 定义LoRA模型配置
lora_config = LoraConfig(
    r=8,  # 矩阵分解的秩，通常在 4~16之间
    lora_alpha=16,  # 系数
    lora_dropout=0.1,  # Dropout 概率
    bias="none",  # 不使用偏置
    task_type=TaskType.SEQ_2_SEQ_LM,  # 任务类型
)

# Step 4: 定义LoRAModel模型
class LoRAModel(nn.Module):
    def __init__(self, model_name='EleutherAI/gpt-neo-2.7B', lora_config=None):
        super(LoRAModel, self).__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name)
        self.lora_model = get_peft_model(self.transformer, lora_config)
        self.fc = nn.Linear(self.transformer.config.n_embd, 1)  # 输出为一个数值

    def forward(self, input_ids, attention_mask=None):
        outputs = self.lora_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, -1, :]  # 使用最后一个 token 的输出作为表示
        return self.fc(pooled_output)

# Step 5: 加载训练好的模型
model = LoRAModel()
model.load_state_dict(torch.load('../models/lora_reward_model.pth'))  # 加载训练好的模型

# Step 6: 设置模型为评估模式
model.eval()

# Step 7: 在测试集上进行验证
predictions = []
targets = []

with torch.no_grad():
    for batch in test_loader:
        inputs, targets_batch = batch
        outputs = model(inputs)
        predictions.extend(outputs.squeeze().cpu().numpy())  # 将预测结果转移到CPU并转为numpy数组
        targets.extend(targets_batch.cpu().numpy())  # 将目标值转移到CPU并转为numpy数组

# Step 8: 计算验证集上的损失
mse_loss = nn.MSELoss()
test_loss = mse_loss(torch.tensor(predictions), torch.tensor(targets)).item()

print(f"Test Loss: {test_loss}")

# Step 9: 保存测试集预测结果
test_results = pd.DataFrame({'Prediction': predictions, 'True Reward': targets})
test_results.to_csv('../datas/test_predictions.csv', index=False)
print("测试集预测结果已保存为 'test_predictions.csv'")
