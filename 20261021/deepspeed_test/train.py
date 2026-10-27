import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 指定GPU

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch


#关键点1：DeepSpeed 配置必须在模型创建之前执行。


training_args = TrainingArguments(
    output_dir="/teacher_train/model/DeepSeek_qwen_1.5b/",
    # overwrite_output_dir=True,  #新版本取消了这个参数 默认为覆盖原文件夹
    
    # 训练设置
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,  # 有效 batch size = 2 * 32 = 64
    
    # 学习率
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    
    # 保存策略
    save_strategy="steps",
    save_steps=5,
    save_total_limit=50,

    # 日志
    logging_steps=10,
    report_to="none",

    # 性能优化
    fp16=True,                      # 混合精度训练
    )


# accelerate/torchrun 会为每个进程设置 LOCAL_RANK。若不绑卡，很多 CUDA 分配会默认落到 cuda:0
local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 获取当前进程的编号(local_rank)，用于绑定到对应GPU
torch.cuda.set_device(local_rank)  #根据编号去设置当前进程使用的GPU


#======================= 加载模型和 Tokenizer ========================
model_name = "/teacher_train/model/deepseek_qwen_7b/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        torch_dtype=torch.float16,
    )  #不去指定加载的设备，zero3 不会加载完整模型

# ======================== 数据处理函数（核心）========================
def preprocess_function(examples):
    """
    标准的 SFT 数据处理流程
    - 只训练 assistant 的回答部分
    - 正确处理截断和 EOS token
    - 保持 chat template 格式
    """
    max_length = 1024
    
    input_ids_list = []
    labels_list = []
    
    for question, cot, response in zip(
        examples["Question"], 
        examples["Complex_CoT"], 
        examples["Response"]
    ):
        # 构建对话消息
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
        # apply_chat_template 已经自动添加了 EOS，所以只需要处理截断
        if len(full_ids) > max_length:
            # 截断到 max_length-1，然后手动添加 EOS
            full_ids = full_ids[:max_length - 1]
            full_ids.append(tokenizer.eos_token_id)
        # 如果没有超长，full_ids 已经包含 EOS，不需要额外处理
        
        # ============ Step 5: 构建 labels（关键！）============
        # prompt 部分设为 -100（不计算 loss）
        # assistant 部分保留原值（计算 loss）
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

# ======================== 加载和处理数据集 ========================
dataset = load_dataset(
    "/teacher_train/data/medical-o1-reasoning-SFT/",
    "zh",
    split="train[:20000]"
)

# 划分训练集和验证集
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

# 应用数据处理
processed_dataset = split_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Processing dataset"
)

train_dataset = processed_dataset["train"]

# ======================== LoRA 配置 ========================
lora_config = LoraConfig(
    r=8,                    # LoRA 秩
    lora_alpha=16,           # LoRA alpha
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj"      # MLP（可选）
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# ======================== Data Collator ========================
# 使用 DataCollatorForSeq2Seq（支持自定义 labels）
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
)


# ======================== 创建 Trainer ========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)


trainer.train()







