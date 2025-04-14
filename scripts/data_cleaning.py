import pandas as pd

# 读取已处理的数据
df = pd.read_csv('../datas/train_lmsys_chat_with_rewards_new.csv')

# # 设置 pandas 不截断长文本
# pd.set_option('display.max_colwidth', None)

# # 输出conversation列的前五行完整内容
# print(df['conversation'].head())

df_with_reward = df[df['reward'].notna()]

num_missing_rewards = df['reward'].isna().sum()

# 输出reward为空的行数
print(f"Total conversations with missing rewards: {num_missing_rewards}")

# 输出新的数据总量
print(f"Total conversations with valid rewards: {len(df_with_reward)}")

# 将有reward的记录保存到新的文件中
df_with_reward.to_csv('../datas/train_lmsys_chat_with_rewards_cleaned.csv', index=False)
