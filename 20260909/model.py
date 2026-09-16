"""
    构建模型
"""

import torch
import torch.nn as nn

class CBOWEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_ids):
        embeddings = self.embedding(context_ids)
        mask = (context_ids != 0).unsqueeze(-1)
        embedding_sum = (embeddings * mask).sum(dim=1)
        no_padding = mask.sum(dim=1).clamp_min(1)
        print(no_padding)

        return self.out(embedding_sum / no_padding)


if __name__ == "__main__":
    model = CBOWEmbeddingModel(vocab_size=8, embedding_dim=5, padding_idx=0)

    id = torch.tensor([[1, 2, 0, 3], [2, 1, 0, 4],[3, 2, 0, 0]])

    out = model(id)
    print(out.shape)    #   [3, 8]