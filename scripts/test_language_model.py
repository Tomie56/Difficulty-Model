from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def test_gpt_neo():
    # 加载预训练的 GPT-Neo 模型和分词器
    model_name = "EleutherAI/gpt-neo-2.7B"  
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # 输入文本
    prompt = "Once upon a time, in a land far away,"
    
    # 对输入文本进行编码，转换为模型所需的格式
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 使用模型生成输出
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    
    # 解码输出生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Prompt:", prompt)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    test_gpt_neo()
  
