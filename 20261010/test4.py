#加载分类模型
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("D:/PythonProjects/LLM/model/rbt3",local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("D:/PythonProjects/LLM/model/rbt3",local_files_only=True)


sen = '真术相成真好!'
inputs = tokenizer(sen, return_tensors='pt').to(model.device)

result = model(**inputs)
print(result)


# #获取模型参数
# print(model.config)

# #获取类比标签
# print(model.config.id2label)