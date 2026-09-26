import torch.nn as nn
import torch
import math
import copy


'''
    clones: 克隆
    Embeddings: 词向量
    PositionalEncoding: 位置编码
    attention: 注意力机制
    MultiHeadedAttention: 多头注意力机制
    LayerNorm: 层归一化
    SublayerConnection: 残差连接+层归一化
    PositionwiseFeedForward: 前馈神经网络
    Generator: 生成器
'''


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):

    def __init__(self, vocab_size, d_model):
        """词向量初始化

        Args:
            vocab_size (_type_): 词表大小
            d_model (_type_): 词向量维度
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        '''构建max_len * d_model的矩阵，位置编码矩阵'''
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        '''偶数sin 奇数cos'''
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        '''pe shape: [max_len, d_model] -> [1, max_len, d_model]'''
        '''pe与词向量形状要一致 做加法'''
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    """计算注意力权重

    Args:
        query (_type_): [batch_size, n_head, seq_len, d_k]
        key (_type_): [batch_size, n_head, seq_len, d_k]
        value (_type_): [batch_size, n_head, seq_len, d_v]
        mask (_type_, optional): 掩码. Defaults to None.
        dropout (_type_, optional): Dropout概率. Defaults to None.

    Returns:
        _type_: _description_
    """
    d_k = query.size(-1)
    '''计算注意力权重 scores = Q * K^T / sqrt(d_k)'''
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """多头注意力机制

        Args:
            query (_type_): [batch_size, seq_len, d_model]
            key (_type_): [batch_size, seq_len, d_model]
            value (_type_): [batch_size, seq_len, d_model]
            mask (_type_, optional): 掩码. Defaults to None.

        Returns:
            _type_: _description_
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        '''view的作用是将输入张量的形状改变为指定的形状，且不改变其数据内容。'''
        '''也就是逻辑上将8个头分开了'''
        '''view:   [batch_size, seq_len, d_model] 
                -> [batch_size, seq_len, h, d_k] 
                -> [batch_size, h, seq_len, d_k]'''
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        '''x.transpose(1, 2)的shape: [batch_size, h, seq_len, d_k] -> [batch_size, seq_len, h, d_k]'''
        '''contiguous()的作用是将输入张量在内存中变为连续的存储方式，以便后续的操作可以更高效地进行。'''
        '''view的作用是将输入张量的形状改变/还原为[batch_size, seq_len, d_model]，且不改变其数据内容。'''
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """层归一化"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """残差连接+层归一化"""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """sublayer: 子层函数"""
        '''Pre-LN: 先归一化再残差连接'''
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class Generator(nn.Module):
    """生成器"""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)