"""
Text Vectorization

One-hot:（不使用  只看过程）
    维度爆炸
    正交的，相似度全为0，无法区分近义词（语义鸿沟）

    
1.分词
2.去重后，构建词汇表
3.构建 word2index 和 index2word
    word2index: {word: index}   --- 输入模型
    index2word: {index: word}   --- 用来做模型输出的匹配
4.获取词向量
    使用word2index中，word对应的index值转成词向量

"""

import jieba
import collections
import torch



with open("20260904/data/info.txt", "r", encoding="utf-8") as f:
    vocab = ['PAD']
    tokens = [jieba.lcut(line.strip()) for line in f.readlines()]
    ts = [word for t in tokens for word in t]
    counter = collections.Counter(ts)
    # print(counter)

    for word in counter:
        vocab.append(word)

    #   构建w2i和i2w
    word2index = {w:i for i,w in enumerate(vocab)}
    index2word = {i:w for i,w in enumerate(vocab)}


    print(vocab)
    print(word2index)
    print(index2word)


def create_onehot(word):
    #   onehot维度就是词表大小
    onehot = torch.zeros(len(vocab))
    index = word2index[word]
    onehot[index] = 1
    return onehot

if __name__ == "__main__":
    print(create_onehot("喜欢"))




