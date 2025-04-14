# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_response(prompt, model, tokenizer, max_length=50, num_return_sequences=1):
    # 对输入的prompt进行编码
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # 使用模型生成响应
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 解码生成的输出
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses

# 3. 测试生成函数
prompt = "Hello, how are you today?"
responses = generate_response(prompt, model, tokenizer, max_length=50, num_return_sequences=1)

# 打印生成的响应
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}")
