import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第一个 GPU
'''
QLora量化训练

1.准备量化工具
2.量化模型
3.准备数据集
4.给模型配置lora的矩阵
5.配置训练参数
6.训练模型
7.保存模型
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments


#1.准备量化工具

from transformers import BitsAndBytesConfig
import torch

#配置4bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

#2.量化模型
model_name="/mnt/workspace/model/qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=True,
    device_map={"": 0},
    torch_dtype=torch.float16,  # 替代 torch_dtype
    quantization_config=bnb_config
)




tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/model/qwen2.5-7B-Instruct")

# print(model)


# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("/root/LLM/20261015/medical-o1-reasoning-SFT/", "zh",split="train[:]")

split_dataset = ds.train_test_split(test_size=0.2, seed=42)

def preprocess_function(examples):

    input_ids_list = []
    labels_list = []
    max_length = 1024
    for question, cot, response in zip(
        examples["Question"], 
        examples["Complex_CoT"], 
        examples["Response"]
    ):

        messages = [
                {
                    "role": "system", 
                    "content": "您是一位在临床推理、诊断和治疗计划方面具有高级知识的医学专家。请回答以下医学问题。"
                },
                {
                    "role": "user", 
                    "content": question
                },
                {
                    "role": "assistant", 
                    "content": f"<think>\n{cot}\n</think>\n{response}"
                }
            ]
        # ============ Step 1: 生成完整对话 ============
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # ============ Step 2: 生成 prompt 部分（不含 assistant 回复）============

        prompt_messages = messages[:-1]  # 去掉最后的 assistant

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True  # 添加 assistant 开始标记
        )

        # ============ Step 3: Tokenize ============
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)


        # ============ Step 4: 处理截断（保留 EOS）============
        #资源够尽量不阶段
        if len(full_ids) > max_length:
            # 截断到 max_length-1，然后手动添加 EOS
            full_ids = full_ids[:max_length - 1]
            full_ids.append(tokenizer.eos_token_id)
        # 如果没有超长，full_ids 已经包含 EOS，不需要额外处理

        # ============ Step 5: 构建 labels（关键！）============

        prompt_len = len(prompt_ids)

        labels = (
            [-100] * prompt_len +           # prompt 部分：不训练
            full_ids[prompt_len:]            # assistant 部分：训练（包含 EOS）
        )

        # 确保长度一致
        labels = labels[:len(full_ids)]
        
        input_ids_list.append(full_ids)
        labels_list.append(labels)
    return {
        "input_ids": input_ids_list,
        "attention_mask": [[1] * len(ids) for ids in input_ids_list],
        "labels": labels_list
    }


processed_dataset = split_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=ds.column_names,
)




# ======================== 4. LoRA 配置 ========================

#配置lora 矩阵的参数  已经添加矩阵的位置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


lora_model = get_peft_model(model, lora_config)
# print(lora_model)


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
)

# 开启梯度检查点（节省显存）
model.enable_input_require_grads()


training_args = TrainingArguments(
    output_dir="/root/LLM/Qmodels",

    # 训练设置
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # 有效 batch size = 2 * 8 = 16

    # 学习率
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=50,

    # 正则化和梯度
    weight_decay=0.01,
    max_grad_norm=1.0,

    # 日志
    logging_strategy="steps",
    logging_steps=50,

    # 保存策略
    save_strategy="epoch",
    save_total_limit=2,

    # 评估策略
    eval_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,

    # 梯度检查点
    gradient_checkpointing=True,

    # 混合精度
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),

    # 优化器
    optim="adamw_torch_fused",

    report_to="none",
)



trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    data_collator=data_collator,
)


if __name__ == "__main__":
    trainer.train()

    trainer.save_model("/root/LLM/qwen2.5-7B-QLoRA")


