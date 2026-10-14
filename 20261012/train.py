import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

import evaluate
import numpy as np

dataset = load_dataset("/root/LLM/20261012/nlpcc_2017", split="train", cache_dir="/root/LLM/20261012/cache")

dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(dataset['train'].column_names)

tokenizer = T5TokenizerFast.from_pretrained("/root/model/mengzi-t5-base", local_files_only=True)
tokenizer.save_pretrained("/root/model/mengzi-t5-base")

model = AutoModelForSeq2SeqLM.from_pretrained("/root/model/mengzi-t5-base", local_files_only=True)

def preprocess_function(examples):
    '''遍历列表中的每个元素，提取content和title字段'''
    contents = ["摘要生成：\n" + e['content'] for e in examples["data"]]
    inputs = tokenizer(contents, max_length=512, truncation=True)


    labels = tokenizer(text_target = [e['title'] for e in examples["data"]], max_length=64, truncation=True)

    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)





rouge = evaluate.load("rouge")
def compute_metrics(eval_pred):
    """
    1.Trainer遍历eval_pred中的每个元素，提取预测值和标签
    2.收集所有预测值和标签
    3.调用compute_metrics函数，返回ROUGE指标
    """

    pre, label = eval_pred
    pre = np.asarray(pre)
    if pre.ndim == 3:
        pre = pre.argmax(axis=-1)
    pre = np.where(
        (pre >= 0) & (pre < tokenizer.vocab_size),
        pre,
        tokenizer.pad_token_id,
    ).astype(np.int64)
    '''解码'''
    decoded_preds = tokenizer.batch_decode(pre, skip_special_tokens=True)
    labels = np.where(label != -100, label, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    '''分词处理'''
    decoded_preds = [" ".join(p) for p in decoded_preds]
    decoded_labels = [" ".join(l) for l in decoded_labels]

    scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    return {
        "rouge1": scores["rouge1"].mid.fmeasure * 100,
        "rouge2": scores["rouge2"].mid.fmeasure * 100,
        "rougeL": scores["rougeL"].mid.fmeasure * 100,
    }


args = Seq2SeqTrainingArguments(
    output_dir="/root/LLM/20261012/output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=16,
    logging_steps=8,
    predict_with_generate=True,
    generation_max_length=64,
    num_train_epochs=10,
    save_total_limit=3,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    metric_for_best_model="rougeL",
    load_best_model_at_end=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)

print(f"Training device: {trainer.args.device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(trainer.args.device)}")

trainer.train()

trainer.save_model("/root/LLM/20261012/title/best_model")
