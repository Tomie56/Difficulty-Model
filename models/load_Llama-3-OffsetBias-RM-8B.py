# import os

# os.environ["TRANSFORMERS_CACHE"] = "D:\Transformers_cache" 

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "NCSOFT/Llama-3-OffsetBias-RM-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(model)

# def generate_response(prompt, model, tokenizer, max_length=100, num_return_sequences=1):

#     inputs = tokenizer(prompt, return_tensors="pt")
    
#     outputs = model.generate(
#         inputs['input_ids'],
#         max_length=max_length,
#         num_return_sequences=num_return_sequences,
#         no_repeat_ngram_size=2, 
#         pad_token_id=tokenizer.eos_token_id 
#     )
    
#     responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return responses

# prompt = "What is the capital of France?"
# responses = generate_response(prompt, model, tokenizer, max_length=100, num_return_sequences=1)
# print(responses)
