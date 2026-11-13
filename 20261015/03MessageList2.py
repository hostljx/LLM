import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第一个 GPU
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/teacher_train/model/DeepSeek_qwen_1.5b/")

model = AutoModelForCausalLM.from_pretrained("/teacher_train/model/DeepSeek_qwen_1.5b/",device_map="auto")


#消息列表
messages = [{"role": "system", "content": "按照json格式输出答案"},
            {"role": "user", "content": "你是谁"}]

messages2 = [{"role": "system", "content": "按照json格式输出答案"},
            {"role": "user", "content": "你是谁"},
            {"role": "assistant", "content": "我是谁"}]


input_ids = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=False, 
    tokenize=False,
    return_tensors="pt"
)
print(input_ids)



full_text = tokenizer.apply_chat_template(
    messages2,
    add_generation_prompt=False, 
    tokenize=False,
    return_tensors="pt"
)
print(full_text)



