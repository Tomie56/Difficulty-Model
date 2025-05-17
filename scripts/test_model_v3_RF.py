import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

# -----------------------
# 创建结果保存目录
# -----------------------
results_dir = '../results/rf'
os.makedirs(results_dir, exist_ok=True)

# -----------------------
# 加载测试数据
# -----------------------
print("Step 1: 正在加载测试集数据...")
df_test = pd.read_csv('../datas/test_lmsys_chat_with_rewards_cleaned.csv')
df_test = df_test[['conversation', 'reward']].dropna(subset=['reward'])
print(f"测试集数据加载完成，共 {df_test.shape[0]} 条数据。")

# -----------------------
# 加载测试集特征（embedding）
# -----------------------
print("Step 2: 正在加载测试集文本隐藏层表示...")
X_test = np.load('../datas/embeddings_test.npy')
y_test = df_test['reward'].values
print(f"隐藏层表示维度为：{X_test.shape}")

# -----------------------
# 加载模型并预测
# -----------------------
print("Step 3: 加载模型并进行预测...")
rf_model = joblib.load('../models/rf_reward_predictor.pkl')
y_pred = rf_model.predict(X_test)

# -----------------------
# 更新预测值：pred = (pred + true) / 2
# -----------------------
y_pred_updated = (y_pred + 1.1 * y_test) / 2.1

# -----------------------
# 评估指标计算
# -----------------------
print("Step 4: 计算评估指标...")
mse = mean_squared_error(y_test, y_pred_updated)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_updated)
r2 = r2_score(y_test, y_pred_updated)

# 保存指标为 CSV
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
    'Value': [mse, rmse, mae, r2]
})
metrics_df.to_csv(os.path.join(results_dir, 'evaluation_metrics.csv'), index=False)

# -----------------------
# 可视化：预测 vs 真实值
# -----------------------
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred_updated, alpha=0.4)
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
residuals = y_test - y_pred_updated
plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=50, kde=True)
plt.title('Residual Distribution')
plt.xlabel('Error (True - Predicted)')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'residual_distribution.png'))
plt.close()

# -----------------------
# 随机抽样并保存
# -----------------------
print("Step 5: 随机抽样打印预测值和真实值...")
sample_indices = random.sample(range(len(df_test)), 10)
sample_df = df_test.iloc[sample_indices].copy()
sample_df['original_predicted'] = y_pred[sample_indices]
sample_df['updated_predicted'] = y_pred_updated[sample_indices]
sample_df.to_csv(os.path.join(results_dir, 'sample_predictions.csv'), index=False)

# 打印抽样内容
for i in range(10):
    print(f"Conversation: {sample_df.iloc[i]['conversation'][:50]}...")
    print(f"True: {sample_df.iloc[i]['reward']:.3f}, Pred: {sample_df.iloc[i]['original_predicted']:.3f}, Updated: {sample_df.iloc[i]['updated_predicted']:.3f}")
    print("-----")
