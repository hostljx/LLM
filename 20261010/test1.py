#下载模型到本地
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from transformers import AutoModel,AutoTokenizer

# model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",cache_dir="/teacher_data/wtb/教学文档/大模型教学文档/47期/hug/models")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",cache_dir="/teacher_data/wtb/教学文档/大模型教学文档/47期/hug/models")

# print(model)


#魔塔社区模型下载

# import os
# os.environ['MODELSCOPE_ENDPOINT'] = 'https://www.modelscope.ai'
# #Model Download
# from modelscope import snapshot_download
# model_dir = snapshot_download('Qwen/Qwen2.5-0.5B')


# modelscope download --model Qwen/Qwen2.5-0.5B --local_dir /teacher_data/wtb/教学文档/大模型教学文档/47期/hug/models


