from torch.utils.data import Dataset
# from pathlib import Path
import jieba
from collections import Counter
import torch.nn.functional as F
import torch




class CBOWDataset(Dataset):
    def __init__(self, root, min_freq=1, window_size=2):
        super().__init__()
        self.root = root
        self.vocab = ["<PAD>"]
        self.word2idx = {}
        self.idx2word = {}
        self.min_freq = min_freq
        self.window_size = window_size
        self.__create_vocab()
        self.contexts, self.targets = self.__create_context()

        pass

    def __load_text(self):
        """
        加载语料库

        Yields:
            _type_: _description_
        """        
        with open(self.root, "rt", encoding="utf-8") as f:
            for line in f:
                # 惰性对象
                # 分词

                sentence = jieba.lcut(line.strip())
                yield sentence

    def __create_vocab(self):
        """
        分词 + 构建词汇表
        """        
        counter = Counter()
        for sentence in self.__load_text():
            # 去重
            counter.update(sentence)
            # 构建词汇表
        for token, c in counter.items():
            if c >= self.min_freq:
                self.vocab.append(token)
        self.word2idx = {w:i for i, w in enumerate(self.vocab)}
        self.idx2word = {i:w for i, w in enumerate(self.vocab)}

        # print(self.word2idx)
        # print(self.idx2word)
        # print(counter)


    def __create_context(self):
        contexts = []
        targets = []
        for sentence in self.__load_text():
            # 获取sentence中每个token的id
            token_ids = [self.word2idx[token] for token in sentence]
            PAD_INDEX = self.word2idx["<PAD>"]
            padded_token_ids = [PAD_INDEX] * self.window_size + token_ids + [PAD_INDEX] * self.window_size

            for i, ids in enumerate(token_ids):
                # [i:i+window_size]
                # [i+window_size+1:i+window_size+1+window_size]
                index = i + self.window_size
                # 获取上下文
                context = padded_token_ids[i:index] + padded_token_ids[index+1:index+self.window_size+1]
                contexts.append(context)
                # 保存中心词
                targets.append(token_ids[i])

            # print(token_ids)                
            # print(sentence)
            # print(padded_token_ids)
        return torch.tensor(contexts), torch.tensor(targets)


    def __len__(self):
        return len(self.targets)


    def __getitem__(self, index):
        context = self.context[index]
        target = self.targets[index]
        return context, target


class CBOWDOneHotDataset(CBOWDataset):
    def __getitem__(self, index):
        context = self.contexts[index]
        # one-hot编码
        # 现在context：[0, 0, 2, 3]
        context_onehot = F.one_hot(context, num_classes=len(self.vocab)).float()
        # print(context_onehot)
        target = self.targets[index]
        # 设置PAD的one-hot编码为0
        # context_onehot[context == self.word2idx["<PAD>"]] = 0
        context_onehot[context == 0] = 0

        return context_onehot, target


if __name__ == "__main__":
    # ds = CBOWDataset("data/text.txt", min_freq=1)
    # for x in ds.load_text():
    #     print(x)

    # t = ds.create_context()
    # print(t)

    ds = CBOWDOneHotDataset("data/text.txt", min_freq=1)
    print(ds[0])