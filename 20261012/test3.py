import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只用 0 号卡；要第 1 张就写成 '1'
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

from transformers import Trainer, TrainingArguments

import evaluate

import numpy as np

#加载自己的数据集
data = load_dataset("csv", data_files="/teacher_data/wtb/教学文档/大模型教学文档/large_model/bigmodel/02-huggingface/ChnSentiCorp_htl_all.csv",split = "train")




train_and_test = data.train_test_split(test_size=0.2)  #分割数据集，默认是8:2


data_filter = train_and_test.filter(lambda x: x['review'] is not None)  #过滤掉review为None的数据

model = AutoModelForSequenceClassification.from_pretrained("/teacher_train/model/roberta-base-finetuned-dianping-chinese/")


tokenizer = AutoTokenizer.from_pretrained("/teacher_train/model/roberta-base-finetuned-dianping-chinese/")


def preprocess_function(examples):
    model_input = tokenizer(examples['review'], max_length=128, truncation=True)
    model_input['labels'] = examples['label']

    return model_input


tokenized_data_train = data_filter["train"].map(preprocess_function, batched=True,remove_columns=['review','label'])

tokenized_data_test = data_filter["test"].map(preprocess_function, batched=True,remove_columns=['review','label'])


#数据整理器
collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True,pad_to_multiple_of=8)


train_args = TrainingArguments(
    output_dir='/teacher_data/wtb/教学文档/大模型教学文档/47期/hug-20261012/results',
    evaluation_strategy='steps',
    eval_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    gradient_accumulation_steps=10, 
    metric_for_best_model="accuracy",
    )

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions,axis=-1)

    return accuracy.compute(predictions=predictions, references=labels)

    #必须啊返回字典格式，因为是根据metric_for_best_model="accuracy"来获取的参数


trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_test,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()

    trainer.save_model("/teacher_data/wtb/教学文档/大模型教学文档/47期/hug-20261012/results/best_model")




















