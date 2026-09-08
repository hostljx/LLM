"""使用 nn.Embedding 构建的 CBOW 模型。"""

import torch
from torch import nn


class CBOWModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim=100):
		super().__init__()
		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=embedding_dim,
			padding_idx=0,
		)
		self.out = nn.Linear(embedding_dim, vocab_size)

	def forward(self, context_ids):
		# context_ids 的 shape: (N, 4)，其中 0 表示 PAD
		embeddings = self.embedding(context_ids)
		mask = (context_ids != 0).unsqueeze(-1)
		embedding_sum = (embeddings * mask).sum(dim=1)
		non_pad_count = mask.sum(dim=1).clamp_min(1)
		hidden = embedding_sum / non_pad_count
		return self.out(hidden)


if __name__ == "__main__":
	model = CBOWModel(vocab_size=5, embedding_dim=10)
	context_ids = torch.tensor([[2, 2, 0, 0], [1, 3, 4, 0]])
	print(model(context_ids).shape)
