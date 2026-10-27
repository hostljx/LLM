import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,    
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from peft import LoraConfig,get_peft_model
import torch


model_name="/teacher_train/model/DeepSeek_qwen_1.5b/"


local_rank = int(os.environ.get("LOCAL_RANK", 0)) #获取当前进程的编号(local_rank)，用于绑定到对应GPU
torch.cuda.set_device(local_rank) #根据编号去设置当前进程使用的GPU
#例如你使用rank 0,1,2,3 四个进程，那么local_rank 就是0,1,2,3
#对应第一rank 0绑定cuda:0,rank 1绑定cuda:1,rank 2绑定cuda:2,rank 3绑定cuda:3

#按照设备去加载模型
model = AutoModelForCausalLM.from_pretrained(model_name,
local_files_only=True,
device_map={"": local_rank},#指定模型在对应GPU上进行加载
torch_dtype=torch.bfloat16)


tokenizer = AutoTokenizer.from_pretrained(model_name)


ds = load_dataset("/teacher_train/data/medical-o1-reasoning-SFT/", "zh",split="train[:200]")

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
    r=8,                    # LoRA 秩
    lora_alpha=16,           # LoRA alpha
    target_modules=[
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"# MLP（可选）
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


lora_model = get_peft_model(model, lora_config)
# print(lora_model)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
)


model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

training_args = TrainingArguments(
    output_dir="/teacher_data/wtb/教学文档/大模型教学文档/47期/distributed_traing/ddp_out",#输出目录，用于保存模型和日志
    overwrite_output_dir=True,#覆盖输出目录的现有内容
    num_train_epochs=5,#训练的总轮次
    per_device_train_batch_size=1,#每个训练设备上的批次大小
    save_steps=100,#每隔多少步保存一次模型
    save_total_limit=20,#保存的模型文件数量限制
    prediction_loss_only=True,#在评估期间只返回损失
    report_to="none",  # 关键：启用 swanlab 上报
    logging_dir="./logs",
    save_strategy="steps",
    gradient_accumulation_steps=16, #梯度累积步数   表示训练16个小批次才会更新一次模型参数
    remove_unused_columns=False,  # 是否移除数据集中未使用的列 False：保留所有列
    gradient_checkpointing=True,#启用梯度检查点


    #分布式相关的参数

    ddp_backend="nccl",  # 使用 NCCL 后端（GPU 分布式训练标准）

    gradient_checkpointing_kwargs={"use_reentrant": False},
    #False：非重入式检查点，避免 DDP 冲突,True：重入式检查点，允许 DDP 冲突

    ddp_find_unused_parameters=False, 
    # LoRA 不需要查找未使用参数，必须设置为 False DDP有错的时候可以尝试为True
    #forward/backward 没走到” 的情况（动态分支、条件执行、MoE/门控、可选模块、某些 step 才用到的 loss/头等）。
    #这时如果设成 False，DDP 可能报错（梯度同步卡住/报“unused parameters”相关问题

    #添加混合精度训练（节省内存）
    bf16=True,  # 或者 bf16=True（如果 GPU 支持）
)


trainer = Trainer(
    model=lora_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=processed_dataset["train"],
)

trainer.train()