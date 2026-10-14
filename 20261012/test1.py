import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

from torch.utils.data import DataLoader



#线上去拉数据集
# from datasets import load_dataset

# ds = load_dataset("zeroMN/nlp_corpus_zh")

# print(ds)

#加载自己的数据集
data = load_dataset("csv", data_files="/teacher_data/wtb/教学文档/大模型教学文档/large_model/bigmodel/02-huggingface/ChnSentiCorp_htl_all.csv",split = "train")

#可用的构建器包括："json", "csv", "parquet", "arrow", "text", 
# "xml", "webdataset", "imagefolder", "audiofolder", "videofolder"

# print(data['train'][0]["review"])

# print(data['train']["review"][0])


train_and_test = data.train_test_split(test_size=0.2)  #分割数据集，默认是8:2

# print(train_and_test)


# print(train_and_test["train"].select([1,2,3]))

data_filter = train_and_test.filter(lambda x: x['review'] is not None)  #过滤掉review为None的数据


tokenizer = AutoTokenizer.from_pretrained("/teacher_train/model/roberta-base-finetuned-dianping-chinese/")


def preprocess_function(examples):
    model_input = tokenizer(examples['review'], max_length=128, truncation=True)
    model_input['labels'] = examples['label']

    return model_input


tokenized_data_train = data_filter["train"].map(preprocess_function, batched=True,remove_columns=['review','label'])

tokenized_data_test = data_filter["test"].map(preprocess_function, batched=True,remove_columns=['review','label'])



collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True,pad_to_multiple_of=8)

dl = DataLoader(tokenized_data_train, batch_size=16, collate_fn=collator, shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained("/teacher_train/model/roberta-base-finetuned-dianping-chinese/")

for batch in dl:
    outputs = model(**batch)
    print(outputs.loss)
















