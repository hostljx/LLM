import copy
import random

import torch
import cfg
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import Config, Vocabulary, en_tokenize, load_train, zh_tokenize
from model import Decoder, Encoder


class TranslationDataset(Dataset):
	def __init__(self, data_path, zh_vocab, en_vocab):
		self.samples = []

		for zh_text, en_text in load_train(data_path):
			source = [
				zh_vocab.word2index.get(token, zh_vocab.UNK_INDEX)
				for token in zh_tokenize(zh_text.strip())
			]
			target = [
				en_vocab.SOS_INDEX,
				*[
					en_vocab.word2index.get(token, en_vocab.UNK_INDEX)
					for token in en_tokenize(en_text.strip())
				],
				en_vocab.EOS_INDEX
			]

			if source and len(target) > 1:
				self.samples.append((source, target))

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		return self.samples[index]


def make_collate_fn():
	def collate(batch):
		sources, targets = zip(*batch)
		source_lengths = torch.tensor([len(source) for source in sources])
		source_length = max(map(len, sources))
		target_length = max(map(len, targets))

		source_batch = torch.full(
			(len(batch), source_length),
			Vocabulary.PAD_INDEX,
			dtype=torch.long
		)
		target_batch = torch.full(
			(len(batch), target_length),
			Vocabulary.PAD_INDEX,
			dtype=torch.long
		)

		for row, source in enumerate(sources):
			source_batch[row, :len(source)] = torch.tensor(source)
		for row, target in enumerate(targets):
			target_batch[row, :len(target)] = torch.tensor(target)

		return source_batch, target_batch, source_lengths

	return collate


def train_one_epoch(encoder, decoder, loader, optimizer, criterion, device):
	encoder.train()
	decoder.train()
	total_loss = 0.0

	progress_bar = tqdm(loader, desc="Training", leave=False)
	for source, target, source_lengths in progress_bar:
		source = source.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		encoder_outputs, hidden = encoder(source, source_lengths)

		decoder_input = target[:, :-1]
		expected = target[:, 1:]
		logits, _ = decoder(
			decoder_input,
			hidden,
			encoder_outputs,
			source_lengths
		)

		loss = criterion(
			logits.reshape(-1, logits.size(-1)),
			expected.reshape(-1)
		)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(
			list(encoder.parameters()) + list(decoder.parameters()),
			max_norm=1.0
		)
		optimizer.step()
		total_loss += loss.item()

		if device.type == "cuda":
			allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
			reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
			memory_text = f"{allocated:.0f}/{reserved:.0f} MB"
		else:
			memory_text = "CPU"
		progress_bar.set_postfix(loss=f"{loss.item():.4f}", gpu_mem=memory_text)

	return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(encoder, decoder, loader, criterion, device):
	encoder.eval()
	decoder.eval()
	total_loss = 0.0

	for source, target, source_lengths in loader:
		source = source.to(device)
		target = target.to(device)

		encoder_outputs, hidden = encoder(source, source_lengths)
		decoder_input = target[:, :-1]
		expected = target[:, 1:]
		logits, _ = decoder(
			decoder_input,
			hidden,
			encoder_outputs,
			source_lengths
		)

		loss = criterion(
			logits.reshape(-1, logits.size(-1)),
			expected.reshape(-1)
		)
		total_loss += loss.item()

	return total_loss / max(len(loader), 1)


def main():
	random.seed(42)
	torch.manual_seed(42)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	zh_vocab = Vocabulary.load("word2index.json", "word2index")
	en_vocab = Vocabulary.load("en_word2index.json", "en_word2index")

	dataset = TranslationDataset(Config.FILE_PATH, zh_vocab, en_vocab)
	train_batch_size = 16 if device.type == "cuda" else 64
	train_loader = DataLoader(
		dataset,
		batch_size=train_batch_size,
		shuffle=True,
		collate_fn=make_collate_fn()
	)
	dev_dataset = TranslationDataset(Config.DEV_FILE_PATH, zh_vocab, en_vocab)
	dev_loader = DataLoader(
		dev_dataset,
		batch_size=32 if device.type == "cuda" else 128,
		shuffle=False,
		collate_fn=make_collate_fn()
	)

	encoder = Encoder(len(zh_vocab.vocab), dropout=0.3).to(device)
	decoder = Decoder(len(en_vocab.vocab), dropout=0.3).to(device)

	optimizer = torch.optim.Adam(
		list(encoder.parameters()) + list(decoder.parameters()),
		lr=0.0005
	)
	criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_INDEX)

	max_epochs = 100
	patience = 5
	best_loss = float("inf")
	best_epoch = 0
	best_encoder_state = None
	best_decoder_state = None
	stale_epochs = 0

	for epoch in range(max_epochs):
		loss = train_one_epoch(
			encoder,
			decoder,
			train_loader,
			optimizer,
			criterion,
			device
		)
		val_loss = evaluate(
			encoder,
			decoder,
			dev_loader,
			criterion,
			device
		)
		print(
			f"Epoch {epoch + 1:03d}/{max_epochs}, "
			f"train_loss: {loss:.4f}, val_loss: {val_loss:.4f}"
		)

		if val_loss < best_loss:
			best_loss = val_loss
			best_epoch = epoch + 1
			best_encoder_state = copy.deepcopy(encoder.state_dict())
			best_decoder_state = copy.deepcopy(decoder.state_dict())
			stale_epochs = 0
		else:
			stale_epochs += 1
			if stale_epochs >= patience:
				print(f"Early stopping at epoch {epoch + 1}")
				break

	encoder.load_state_dict(best_encoder_state)
	decoder.load_state_dict(best_decoder_state)

	model_dir = cfg.path / "model"
	model_dir.mkdir(exist_ok=True)
	model_path = model_dir / "translation_model.pth"

	torch.save(
		{
			"encoder": encoder.state_dict(),
			"decoder": decoder.state_dict(),
			"optimizer": optimizer.state_dict(),
			"embedding_size": 256,
			"hidden_size": 256,
			"best_epoch": best_epoch,
			"best_loss": best_loss
		},
		model_path
	)
	print(f"model saved to {model_path} on {device}")


if __name__ == "__main__":
	main()
