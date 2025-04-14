import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler

# 初始化加速器
accelerator = Accelerator()

# 读取数据
train_data = pd.read_csv('../datas/test_lmsys_chat.csv')
train_data = train_data.head(50000)

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# 设置 pad_token
tokenizer.pad_token = tokenizer.eos_token

# 使用加速器来准备模型
model = accelerator.prepare(model)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 初始化混合精度的 GradScaler
scaler = GradScaler()

def generate_response(conversation):
    # 编码输入，减少 max_length
    inputs = tokenizer(conversation, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        # 启用混合精度
        with autocast():
            outputs = model.generate(
                inputs["input_ids"], 
                attention_mask=inputs.get("attention_mask"), 
                max_new_tokens=150, 
                num_return_sequences=1, 
                no_repeat_ngram_size=2, 
                pad_token_id=tokenizer.pad_token_id
            )
    
    # 解码并返回回答
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

responses = []

batch_size = 5  # 根据显存情况调整批次大小
total_rows = len(train_data)

print(f"Total rows to process: {total_rows}")

for i in range(0, total_rows, batch_size):
    print(f"Processing batch {i // batch_size + 1} (rows {i + 1} to {min(i + batch_size, total_rows)})...")  # 打印当前批次处理范围
    batch_conversations = train_data['conversation'][i:i+batch_size]
    batch_responses = []
    
    for j, conversation in enumerate(batch_conversations):
        print(f"  Processing conversation {i + j + 1}/{total_rows}...")  # 打印每个对话的进度
        response = generate_response(conversation)
        batch_responses.append(response)
    
    responses.extend(batch_responses)
    
    print(f"  Finished batch {i // batch_size + 1}. Processed {min(i + batch_size, total_rows)}/{total_rows} rows.")

# 将生成的回答加入数据框
train_data['generated_response'] = responses

print("Saving the results to CSV file...")
train_data.to_csv('../datas/test_lmsys_chat_with_responses.csv', index=False)

print("Processing complete!")
