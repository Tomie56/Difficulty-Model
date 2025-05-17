import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import random

# -----------------------
# 创建保存结果文件夹
# -----------------------
results_dir = '../results/mlp_v1'
os.makedirs(results_dir, exist_ok=True)

# -----------------------
# 定义单层 MLP 模型
# -----------------------
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -----------------------
# 加载测试数据
# -----------------------
df_test = pd.read_csv('../datas/test_lmsys_chat_with_rewards_cleaned.csv')
df_test = df_test[['conversation', 'reward']].dropna(subset=['reward'])

embeddings_test = np.load('../datas/embeddings_test.npy')
embeddings_test = torch.tensor(embeddings_test, dtype=torch.float32)
y_test = torch.tensor(df_test['reward'].values, dtype=torch.float32)

test_dataset = TensorDataset(embeddings_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# -----------------------
# 加载模型
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = embeddings_test.shape[1]
model_mlp = MLPModel(input_size)
model_mlp.load_state_dict(torch.load('../models/mlp_reward_predictor.pth'))
model_mlp.to(device)
model_mlp.eval()

# -----------------------
# 预测
# -----------------------
predictions = []
true_values = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model_mlp(inputs)
        predictions.append(outputs.cpu().numpy())
        true_values.append(labels.cpu().numpy())

predictions = np.concatenate(predictions, axis=0).flatten()
true_values = np.concatenate(true_values, axis=0).flatten()


# -----------------------
# 计算评估指标
# -----------------------
mse = mean_squared_error(true_values, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_values, predictions)
r2 = r2_score(true_values, predictions)

metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
    'Value': [mse, rmse, mae, r2]
})
metrics_df.to_csv(os.path.join(results_dir, 'evaluation_metrics.csv'), index=False)

# -----------------------
# 可视化：预测 vs 真实值
# -----------------------
plt.figure(figsize=(6, 6))
sns.scatterplot(x=true_values, y=predictions, alpha=0.4)
plt.plot([-2, 5], [-2, 5], 'r--')
plt.xlabel('True Reward')
plt.ylabel('Predicted Reward')
plt.title('Predicted vs True Reward')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'predicted_vs_true.png'))
plt.close()

# -----------------------
# 可视化：残差分布
# -----------------------
residuals = true_values - predictions
plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=50, kde=True)
plt.title('Residual Distribution')
plt.xlabel('Error (True - Predicted)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'residual_distribution.png'))
plt.close()

# -----------------------
# 随机抽样输出
# -----------------------
sample_indices = random.sample(range(len(df_test)), 10)
sample_df = df_test.iloc[sample_indices].copy()
sample_df['original_predicted'] = predictions[sample_indices]
sample_df['updated_predicted'] = predictions[sample_indices]
sample_df.to_csv(os.path.join(results_dir, 'sample_predictions.csv'), index=False)

# 打印样本
for i in range(10):
    print(f"Conversation: {sample_df.iloc[i]['conversation'][:50]}...")
    print(f"True: {sample_df.iloc[i]['reward']:.3f}, Pred: {sample_df.iloc[i]['original_predicted']:.3f}, Updated: {sample_df.iloc[i]['updated_predicted']:.3f}")
    print("-----")
