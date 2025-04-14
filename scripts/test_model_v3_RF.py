import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib  # 用于加载保存的模型

# Step 1: 加载测试集数据
print("Step 1: 正在加载测试集数据...")
df_test = pd.read_csv('../datas/test_lmsys_chat_with_rewards_cleaned.csv')

# 只保留需要的列，并去除含有缺失值的行
df_test = df_test[['conversation', 'reward']].dropna(subset=['reward'])
print(f"测试集数据加载完成，共 {df_test.shape[0]} 条数据。")

# Step 2: 加载测试集的预训练文本隐藏层表示
print("Step 2: 正在加载测试集文本隐藏层表示...")
test_embeddings = np.load('../datas/embeddings_test.npy')
print(f"测试集的隐藏层表示加载完成，数据维度为 {test_embeddings.shape}。")

# Step 3: 准备测试数据
X_test = test_embeddings  # 使用测试集文本的隐藏层作为输入特征
y_test = df_test['reward'].values  # 目标是 'reward'
print("Step 3: 测试数据准备完毕，开始进行预测...")

# Step 4: 进行预测
rf_model = joblib.load('../models/rf_reward_predictor.pkl')
y_pred = rf_model.predict(X_test)

print("Step 5: Calculating evaluation metrics...")

# Step 5: 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")

rmse = np.sqrt(mse)  # 均方根误差
print(f"Test RMSE: {rmse}")

mae = mean_absolute_error(y_test, y_pred)  # 平均绝对误差
print(f"Test MAE: {mae}")

r2 = r2_score(y_test, y_pred)  # 决定系数 (R^2)
print(f"Test R²: {r2}")

# Step 6: 打印一部分预测值和真实值进行对比
print("Step 6: Printing sample predictions vs true values...")
for i in range(10):  # 打印前10个预测和真实值
    print(f"Predicted: {y_pred[i]}, True: {y_test[i]}")



