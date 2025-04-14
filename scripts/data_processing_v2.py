import pandas as pd
import re

# 读取原始数据
df = pd.read_csv('../datas/test_lmsys_chat_processed.csv')

# 定义转换函数
def convert_conversation(conversation):
    try:
        # 使用正则表达式替换 'role', 'user', 'assistant', 'content' 两旁的单引号为双引号
        conversation = re.sub(r"('role'|'user'|'assistant'|'content')", lambda match: match.group(0).replace("'", '"'), conversation)
        return conversation
    except Exception as e:
        # 如果处理失败，打印错误信息并返回原数据
        print(f"Error processing conversation: {e}")
        return conversation

# 应用转换函数到 'conversation' 列
df['conversation'] = df['conversation'].apply(convert_conversation)

# 保存处理后的数据到新文件
df.to_csv('../datas/test_lmsys_chat_processed_new.csv', index=False)

print("处理完成，数据已保存至 'train_lmsys_chat_processed_new.csv'.")
