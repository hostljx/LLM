"""
构建CBOW模型

维度：
  一维：[1,2,3]
  二维：[[1,2,3]]
数组的维度： 结构上的维度
词向量的维度： 特征上维度
   [1, 2, 3]  三维的特征
   [1, 2, 3, 4, 5]  五维的特征


请问 W 为什么会是Float?
    pytorch中默认只有叶子节点才能求导；求导需要满足：1. requires_grad=True  2. 数据类型必须是Float
    W是叶子节点

"""
import torch.nn as nn
import torch

class CBOWModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        # embedding: 词嵌入， 也就词向量
        # 这一层训练得到的权重就是 我们需要的 词向量
        # out_features=100: 表示每个token的词向量维度，也就是用100个数字表示一个token的语义信息
        # context_one_hot中的PAD的值全部是0  WX = 0 , 所以不能有bias
        # 即使bias=False， 那么使用Linear训练出来的词向量PAD依然不是全0， 而是有值的。 因为W存在初始化权重
        self.embedding = nn.Linear(in_features=vocab_size, out_features=embedding_dim, bias=False)
        self.out = nn.Linear(in_features=embedding_dim, out_features=vocab_size)


    def forward(self, context_one_hot):
        # context_one_hot的shape: (N, 4, vocab_size)
        # embedding的shape: (N, 4, embedding_dim)
        embedding = self.embedding(context_one_hot)
        # CBOW的原理： 将上下文的词向量求均值后传入下一层
        # 计算embedding的和
        # embedding_sum的shape: (N, embedding_dim)
        embedding_sum = embedding.sum(dim=1)
        # 但是计算均值的时候， PAD要排除，换句话说，就是需要统计非PAD的次数
        # no_pad_count的shape: (N, )  ---> (N, 1)
        no_pad_count = context_one_hot.sum(dim=(1, 2)).view(-1, 1)
        # 计算embedding的均值
        h = embedding_sum / no_pad_count
        # print(h)
        # print(h.shape)
        out = self.out(h)
        return out

if __name__ == '__main__':
    model = CBOWModel(vocab_size=5, embedding_dim=10)
    # (2, 4, 5)
    x = torch.tensor(
        [
            [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        ]
    ).float()
    out = model(x)
    # print(out)
