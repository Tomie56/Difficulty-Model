import pandas as pd
import ast
import re

# 读取原始数据
df = pd.read_csv('../datas/train_lmsys_chat.csv')

# 定义转换函数
def convert_conversation(conversation):
    try:
        # 在缺少逗号的位置添加逗号
        # 通过正则表达式将 'role' 和 'content' 对之间添加逗号
        conversation = re.sub(r"(\}\s*\{)", r"},{", conversation)  # 在两个大括号之间加逗号
        
        # 将处理后的字符串转换为列表
        conversation_list = ast.literal_eval(conversation)
        
        # 重新整理为需要的格式
        return [{"role": item["role"], "content": item["content"]} for item in conversation_list]
    except (SyntaxError, ValueError) as e:
        # 如果解析失败，打印错误信息并返回空列表
        print(f"Error parsing conversation: {e}")
        return []

# 应用转换函数到 'conversation' 列
df['conversation'] = df['conversation'].apply(convert_conversation)

# 保存处理后的数据到新文件
df.to_csv('../datas/train_lmsys_chat_processed.csv', index=False)

print("处理完成，数据已保存至 'test_lmsys_chat_processed.csv'.")
