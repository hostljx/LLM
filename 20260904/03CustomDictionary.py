import jieba



sentence = "真术相成是一家人工智能科技公司"
# 自定义词典
jieba.load_userdict("20260904/data/custom_dict.txt")
tokens = jieba.lcut(sentence)
print(tokens)