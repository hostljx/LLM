import jieba.posseg as pseg


# 词性标注
words = pseg.lcut("我来到北京清华大学")
print(words)