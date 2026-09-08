from dataset import CBOWOneHotDataset
from model import CBOWModel
from torch.utils.data import DataLoader
from torch import nn
import torch


def train():
    # 加载数据集
    dataset = CBOWOneHotDataset("data/text.txt")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # 创建模型训练需要的各种对象
    vocab_size = len(dataset.vocab)
    model = CBOWModel(vocab_size=vocab_size, embedding_dim=256)
    # ignore_index: 用来忽略对应索引值的损失
    # PAD的索引是0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 300
    init_loss = torch.inf  # 初始损失
    # 训练轮次
    for epoch in range(epochs):
        total_loss = 0 # 保存总损失
        for contexts, targets in dataloader:
            # contexts的shape: (N, 4, vocab_size)
            # logits的shape: (N, vocab_size)
            logits = model(contexts)
            # targets的shape：(N，)
            loss = loss_fn(logits, targets)
            total_loss += loss
            # BP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 训练完一轮后输出训练损失
        avg_loss = total_loss / len(dataloader)
        print("Epoch: {}, Loss: {}".format(epoch, avg_loss))

        # 保存词向量
        if avg_loss < init_loss:
            init_loss = avg_loss
            # 保存词向量, 词向量文件保存的数据是：
            # 第一行： 词表大小   词向量维度
            token_embedding = model.embedding.weight.T.detach().numpy()
            vocab_size, embbedding_dim = token_embedding.shape
            # 从第二行开始： token   词向量
            with open("word2vec.txt", "wt", encoding="utf-8") as f:
                f.write(f"{vocab_size} \t {embbedding_dim}\n")
                # 留下来的作业
                # f.write(f"{token_embedding}\n")
                # f.write(f"{2}\t {1111111111111111111111111111111111}\n")




if __name__ == '__main__':
    train()
