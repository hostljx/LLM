from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model = AutoModelForCausalLM.from_pretrained("/teacher_train/model/DeepSeek_qwen_1.5b/",torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("/teacher_train/model/DeepSeek_qwen_1.5b/")

model = PeftModel.from_pretrained(model, "/teacher_data/wtb/教学文档/大模型教学文档/47期/lora-20261015/models/checkpoint-25")
model = model.merge_and_unload()
model.save_pretrained("/teacher_data/wtb/教学文档/大模型教学文档/47期/lora-20261015/models/add_model")
tokenizer.save_pretrained("/teacher_data/wtb/教学文档/大模型教学文档/47期/lora-20261015/models/add_model")

