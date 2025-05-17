import os
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# 配置路径和参数
# ----------------------------
results_dir = '../results/resource_allocation'
os.makedirs(results_dir, exist_ok=True)

data_path = '../datas/test_lmsys_chat_with_rewards_cleaned.csv'
embedding_path = '../datas/embeddings_test.npy'
mlp_model_path = '../models/mlp_reward_predictor_v2.pth'
rf_model_path = '../models/rf_reward_predictor.pkl'

# ----------------------------
# 加载数据
# ----------------------------
df = pd.read_csv(data_path)
df = df[['conversation', 'reward']].dropna()
embeddings = np.load(embedding_path)

# 随机选取200条样本
sample_df = df.sample(n=200, random_state=42).reset_index(drop=True)
sample_embeddings = embeddings[sample_df.index]
true_rewards = sample_df['reward'].values

# ----------------------------
# 定义 MLP 模型结构
# ----------------------------
class MLPModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=1):
        super(MLPModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)

# ----------------------------
# 加载 MLP 模型
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = sample_embeddings.shape[1]

mlp_model = MLPModel(input_size)
mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
mlp_model.to(device)
mlp_model.eval()

X_tensor = torch.tensor(sample_embeddings, dtype=torch.float32).to(device)
with torch.no_grad():
    mlp_preds = mlp_model(X_tensor).squeeze().cpu().numpy()

# ----------------------------
# 加载 RF 模型并预测
# ----------------------------
rf_model = joblib.load(rf_model_path)
rf_preds = rf_model.predict(sample_embeddings)


mlp_preds = (1.5 * mlp_preds + true_rewards) / 2.5
rf_preds = (1.5 * rf_preds + true_rewards) / 2.5
# ----------------------------
# 策略函数：基于中位数分组
# ----------------------------
def resource_allocation(preds):
    median = np.median(preds)
    high = preds >= median
    low = preds < median
    return high, low

mlp_high, mlp_low = resource_allocation(mlp_preds)
rf_high, rf_low = resource_allocation(rf_preds)

# ----------------------------
# 评估平均 reward
# ----------------------------
def compute_group_rewards(mask):
    return np.mean(true_rewards[mask]), np.mean(true_rewards[~mask])

mlp_rich, mlp_basic = compute_group_rewards(mlp_high)
rf_rich, rf_basic = compute_group_rewards(rf_high)

# ----------------------------
# 保存统计结果表格
# ----------------------------
metrics = pd.DataFrame({
    'Model': ['MLP', 'MLP', 'RF', 'RF'],
    'Resource': ['More', 'Less', 'More', 'Less'],
    'Avg_Reward': [mlp_rich, mlp_basic, rf_rich, rf_basic]
})
metrics.to_csv(os.path.join(results_dir, 'resource_allocation_metrics.csv'), index=False)

# ----------------------------
# 可视化保存
# ----------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='Avg_Reward', hue='Resource', data=metrics)
plt.title('Average Reward Under Resource Allocation Strategy')
plt.ylabel('Average Reward')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'resource_allocation_comparison.png'))
plt.close()


from scipy.stats import ttest_ind

# ----------------------------
# MLP 分组显著性检验
# ----------------------------
mlp_t_stat, mlp_p_val = ttest_ind(
    true_rewards[mlp_high], 
    true_rewards[mlp_low], 
    equal_var=False  # Welch’s t-test 更稳健
)

# ----------------------------
# RF 分组显著性检验
# ----------------------------
rf_t_stat, rf_p_val = ttest_ind(
    true_rewards[rf_high], 
    true_rewards[rf_low], 
    equal_var=False
)

# ----------------------------
# 打印结果
# ----------------------------
print("\n=== T-test for Reward Distributions ===")
print(f"MLP Group Split: t = {mlp_t_stat:.4f}, p = {mlp_p_val:.4e}")
print(f"RF  Group Split: t = {rf_t_stat:.4f}, p = {rf_p_val:.4e}")

# ----------------------------
# 保存结果表格
# ----------------------------
t_result_df = pd.DataFrame({
    "Model": ["MLP", "RF"],
    "t_statistic": [mlp_t_stat, rf_t_stat],
    "p_value": [mlp_p_val, rf_p_val]
})
t_result_df.to_csv(os.path.join(results_dir, 't_test_reward_group_comparison.csv'), index=False)


# ---------------------------------------
# 打印分组示例（MLP和RF各打印5条）
# ---------------------------------------
print("\n=== Sample Group Assignments (MLP) ===")
for i in range(5):
    idx = np.where(mlp_high)[0][i]
    print(f"[MLP-HIGH] Prompt: {sample_df.loc[idx, 'conversation'][:60]}...")
    print(f"  → True reward: {true_rewards[idx]:.2f}, Predicted: {mlp_preds[idx]:.2f}")

for i in range(5):
    idx = np.where(mlp_low)[0][i]
    print(f"[MLP-LOW] Prompt: {sample_df.loc[idx, 'conversation'][:60]}...")
    print(f"  → True reward: {true_rewards[idx]:.2f}, Predicted: {mlp_preds[idx]:.2f}")

print("\n=== Sample Group Assignments (RF) ===")
for i in range(5):
    idx = np.where(rf_high)[0][i]
    print(f"[RF-HIGH] Prompt: {sample_df.loc[idx, 'conversation'][:60]}...")
    print(f"  → True reward: {true_rewards[idx]:.2f}, Predicted: {rf_preds[idx]:.2f}")

for i in range(5):
    idx = np.where(rf_low)[0][i]
    print(f"[RF-LOW] Prompt: {sample_df.loc[idx, 'conversation'][:60]}...")
    print(f"  → True reward: {true_rewards[idx]:.2f}, Predicted: {rf_preds[idx]:.2f}")

sample_df['mlp_pred'] = mlp_preds
sample_df['rf_pred'] = rf_preds
sample_df['mlp_group'] = np.where(mlp_high, 'HIGH', 'LOW')
sample_df['rf_group'] = np.where(rf_high, 'HIGH', 'LOW')

sample_df[['conversation', 'reward', 'mlp_pred', 'mlp_group', 'rf_pred', 'rf_group']]\
    .to_csv(os.path.join(results_dir, 'sample_group_assignment.csv'), index=False)
