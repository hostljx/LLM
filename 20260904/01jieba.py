import jieba



with open('20260904/data/info.txt', 'r', encoding='utf-8') as f:
    text = f.readlines()
    for line in text:
        words = list(jieba.cut(line.strip()))
        print(words)



# tokens = jieba.cut("我来到北京清华大学,老登", HMM=False)
# print(list(tokens))


# tokens = jieba.lcut("我来到北京清华大学,老登", HMM=True, cut_all=False)
# print(tokens)

#   cut_all: 全模式


#   搜索引擎模式
tokens = jieba.lcut_for_search("我来到北京清华大学,老登")
print(tokens)