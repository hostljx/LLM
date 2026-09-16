import torch.nn as nn
import torch


"""
    input_size: 输入数据的特征数量
    hidden_size: 隐藏层的大小
    num_layers: RNN的层数
    nonlinearity: 'tanh' 默认
    batch_first: True 输入数据的第一个维度是batch_size
    bidirectional: True 双向RNN

    dropout: 只作用于层之间而非时间步之间
"""

embedding =  nn.Embedding(num_embeddings=100, embedding_dim=10)


rnn = nn.RNN(input_size=10, 
       hidden_size=20, 
       num_layers=1, 
       nonlinearity='tanh', 
       batch_first=True, 
       dropout=0, 
    #  bidirectional=True
       bidirectional=False) # ==> 双向/单向：num_directions = 2/1


x = torch.randint(0, 100, (32, 8))  # 假设输入是一个32个样本，每个样本长度为10的序列

#   vector形状：（batch_size, sequence_length, embedding_dim）
vec = embedding(x)  # 将输入数据进行嵌入    shape: (32, 8, 10)
output, hidden = rnn(vec)  # 将嵌入后的数据输入到RNN中

'''输出的形状为 (batch_size, sequence_length, hidden_size * num_directions)'''
print("output shape:", output.shape)  # [32, 8, 40]  因为是双向20*2

'''隐藏状态的形状为 (num_layers * num_directions, batch_size, hidden_size)'''
print("hidden shape:", hidden.shape)  # [2, 32, 20]

'''表示最后一层所有时间步的隐状态的值'''
res01 = output[:, -1, :]
'''表示所有层最后一个时间步的值'''
res02 = hidden[-1, :, :]
print("res01 shape:", res01.shape)
print("res02 shape:", res02.shape)
print(res01 == res02)  # True

