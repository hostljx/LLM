"""
定义数据集: 上下文  中心词

1. 加载语料库
2. 分词，构建词汇表和id映射
    构建词汇表不要忘记 <PAD>
    <PAD>: 仅仅是一个占位符，没有任何实际意义

    <UNK>: UNKnown   表示未登录词， 少见词

    设定一个最小词频， 将低频数据过滤
    原因： 因为低频数据模型无法得到模型的充分训练，得到的词向量也是不准确的。


训练： 使用 我爱你    训练得到了 词向量模型A
推理： 我恨你   此时的恨 就应该返回  UNK的词向量
此时UNK的词向量有两种方式获取：
1. 随机给一个词向量   不推荐
    缺点：
       a. 随机的词向量和训练得到的词向量分布可能不一致
       b. 分布不一致相当于就是给模型引入了噪声
2. 使用整个句子所有的词向量的均值作为UNK的词向量  推荐



定义函数的时候要遵守： 单一职责原则
也就是一个函数只完成一个功能就可以
"""

from torch.utils.data import Dataset
from collections import Counter
import torch.nn.functional as F
import jieba
import torch


class CBOWDataset(Dataset):
    def __init__(self, root, min_freq=1, window_size=2):
        super().__init__()
        self.root = root
        self.min_freq = min_freq
        self.window_size = window_size
        self.vocab = ["<PAD>"]
        self.word2index = {}
        self.index2word = {}
        self.__create_vocab_index()
        self.context, self.targets = self.create_context()
        pass

    def __load_text(self):
        """
        加载语料库， 逐行返回数据
        :return:
        """
        with open(self.root, 'rt', encoding="UTF-8") as f:
            for line in f:
                # 返回的是生成器对象， 也就是惰性对象
                # 分词
                # sentence = ['自然语言', '处理', '是', '人工智能', '的', '重要', '方向']
                sentence = jieba.lcut(line.strip())
                yield sentence

    def __create_vocab_index(self):
        """
        创建词汇表及其id映射
        :return:
        """
        # 相当于初始化一个字典
        counter = Counter()
        for sentence in self.__load_text():
            # 将token去重
            counter.update(sentence)
        # 构建词汇表
        for token, c in counter.items():
            # 将token存入词汇表
            if c >= self.min_freq:
                self.vocab.append(token)
        # 构建word2index 和 index2word
        self.word2index = {w: i for i, w in enumerate(self.vocab)}
        self.index2word = {i: w for i, w in enumerate(self.vocab)}

    def create_context(self):
        contexts = []
        targets = []
        # 需要得到的是 分词后的列表
        for sentence in self.__load_text():
            # sentence: ['自然语言', '处理', '是', '人工智能', '的', '重要', '方向']
            # 将sentence中每个token的id获取出来
            # token_ids = [1, 2, 3, 4, 5, 6, 7]
            token_ids = [self.word2index[token] for token in sentence]
            # 需要在token_ids的前后添加 window_size个PAD
            PAD_INDEX = self.word2index["<PAD>"]
            padded_token_ids = [PAD_INDEX] * self.window_size + token_ids + [PAD_INDEX] * self.window_size
            # 使用滑动窗口得到上下文和中心词
            for i, ids in enumerate(token_ids):
                # 上文：[i:i+window_size]
                # 下文：[i+window_size+1:i+window_size+1+window_size]
                index = i + self.window_size
                # 获取上下文
                context = padded_token_ids[i:index] + padded_token_ids[index + 1:index + self.window_size + 1]
                contexts.append(context)
                # 保存中心词
                targets.append(token_ids[i])
        return torch.tensor(contexts), torch.tensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        context = self.context[idx]
        target = self.targets[idx]
        return context, target


class CBOWOneHotDataset(CBOWDataset):

    def __getitem__(self, idx):
        context = self.context[idx]
        context = torch.tensor(context)
        # context中的id设置成onehot编码
        # context: [0, 0, 2, 3]
        # one-hot --> 输入给模型的  就是我们 y = WX + b 中的 X
        # 模型的时候 是通过 WX + b 得到的y 和真实的标签做损失
        # 此时PAD的one-hot： [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #          0, 0, 0, 0, 0, 0, 0, 0]
        # PAD是无意义， 不应该对模型有任何的贡献， 所以PAD的onehot应该全部是0才对
        context_one_hot = F.one_hot(context, num_classes=len(self.vocab)).float()
        # print(context_one_hot)
        target = self.targets[idx]
        # context == 0 判断哪些是PAD
        # 将PAD的onehot设置成0
        context_one_hot[context == 0] = 0
        return context_one_hot, target


if __name__ == '__main__':
    ds = CBOWOneHotDataset("data/text.txt", min_freq=1)
    print(ds[0])
