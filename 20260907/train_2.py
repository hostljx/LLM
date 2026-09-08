from torch import nn
from torch.utils.data import DataLoader
import torch

from dataset import CBOWDataset
from model_2 import CBOWModel


def train():
	dataset = CBOWDataset("data/text.txt")
	dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

	vocab_size = len(dataset.vocab)
	model = CBOWModel(vocab_size=vocab_size, embedding_dim=256)
	loss_fn = nn.CrossEntropyLoss(ignore_index=0)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	epochs = 300
	best_loss = torch.inf
	for epoch in range(epochs):
		total_loss = 0.0
		for contexts, targets in dataloader:
			logits = model(contexts)
			loss = loss_fn(logits, targets)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

		avg_loss = total_loss / len(dataloader)
		print("Epoch: {}, Loss: {}".format(epoch, avg_loss))

		if avg_loss < best_loss:
			best_loss = avg_loss
			token_embedding = model.embedding.weight.detach().cpu().numpy()
			vocab_size, embedding_dim = token_embedding.shape
			with open("word2vec_2.txt", "wt", encoding="utf-8") as file:
				file.write(f"{vocab_size} \t {embedding_dim}\n")
				for index, token in enumerate(dataset.vocab):
					vector = " ".join(str(value) for value in token_embedding[index])
					file.write(f"{token} {vector}\n")


if __name__ == "__main__":
	train()
