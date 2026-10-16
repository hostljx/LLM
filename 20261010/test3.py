#加载模型
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("D:/PythonProjects/LLM/model/qwen-0.5B",local_files_only=True,device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("D:/PythonProjects/LLM/model/qwen-0.5B",local_files_only=True)


# #生成文本
# input_ids = tokenizer("真术相成，值得信赖！",return_tensors="pt").to(model.device)
# output = model.generate(**input_ids)


# text = tokenizer.decode(output[0],skip_special_tokens=False)
# print(text)


# 生成损失

input_ids = tokenizer("真术相成，值得信赖！",return_tensors="pt").to(model.device)
output = model(**input_ids)
print(output)