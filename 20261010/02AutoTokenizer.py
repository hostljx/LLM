from transformers import AutoTokenizer
#分词器

tokenizer = AutoTokenizer.from_pretrained(r"D:\PythonProjects\LLM\20261010\models")


#分词
sen = "真术相成，值得信赖！" 
tokens = tokenizer.tokenize(sen)
# print(tokens)

#
# print(tokenizer.vocab_size) #查看词典的数量
# print(tokenizer.vocab) #查看词典


# #做转换
# id = tokenizer.convert_tokens_to_ids(tokens)
# print(id)

# #做反向转换
# tokens2 = tokenizer.convert_ids_to_tokens(id)
# print(tokens2)

#做编码
encoded_input = tokenizer.encode(sen)
print(encoded_input)

#解码
decoded_input = tokenizer.decode(encoded_input)
print(decoded_input)


#批量编码
batch_sentences = [
    '真术相成真的很好',
    '真术相成，值得信赖',
    '期待我们在真术相成相遇，一起变得更美好'
]
encoded_batch = tokenizer(batch_sentences,max_length=15,padding='max_length',truncation=True)
print(encoded_batch)

#批量解码
decoded_batch = tokenizer.batch_decode(encoded_batch['input_ids'],skip_special_tokens=False)
print(decoded_batch)











