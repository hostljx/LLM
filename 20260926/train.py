import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from components import Embeddings, Generator, MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward
from model import Decoder, DecoderLayer, Encoder, EncoderDecoder, EncoderLayer


DEFAULT_DATA_DIR = Path(r"D:\PythonProjects\LLM\20260921\raw")


def load_vocab(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)["word2index"]


def tokenize_english(text):
    return re.findall(r"[A-Za-z]+|'[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]", text)


def tokenize_chinese(text, vocabulary):
    words = sorted(
        (word for word in vocabulary if not word.startswith("<")),
        key=len,
        reverse=True,
    )
    tokens = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        match = next((word for word in words if text.startswith(word, position)), None)
        if match is None:
            tokens.append(text[position])
            position += 1
        else:
            tokens.append(match)
            position += len(match)
    return tokens


def read_parallel_data(data_dir, en_vocab, zh_vocab, max_length):
    pairs = []
    with (data_dir / "cmn.txt").open("r", encoding="utf-8") as file:
        for line in file:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            source_tokens = tokenize_english(fields[0])
            target_tokens = tokenize_chinese(fields[1], zh_vocab)
            if not source_tokens or not target_tokens:
                continue
            if len(source_tokens) + 2 > max_length or len(target_tokens) + 2 > max_length:
                continue
            source_ids = [en_vocab["<SOS>"]]
            source_ids.extend(en_vocab.get(token, en_vocab["<UNK>"]) for token in source_tokens)
            source_ids.append(en_vocab["<EOS>"])
            target_ids = [zh_vocab["<SOS>"]]
            target_ids.extend(zh_vocab.get(token, zh_vocab["<UNK>"]) for token in target_tokens)
            target_ids.append(zh_vocab["<EOS>"])
            pairs.append((source_ids, target_ids))
    if not pairs:
        raise RuntimeError(f"No usable sentence pairs found in {data_dir / 'cmn.txt'}")
    return pairs


class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def make_collate_fn(source_pad, target_pad):
    def collate(batch):
        source_length = max(len(source) for source, _ in batch)
        target_length = max(len(target) for _, target in batch)
        sources = torch.full((len(batch), source_length), source_pad, dtype=torch.long)
        targets = torch.full((len(batch), target_length), target_pad, dtype=torch.long)
        for index, (source, target) in enumerate(batch):
            sources[index, : len(source)] = torch.tensor(source)
            targets[index, : len(target)] = torch.tensor(target)
        return sources, targets

    return collate


def subsequent_mask(size, device):
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))


def build_model(source_vocab_size, target_vocab_size, d_model, heads, layers, d_ff, dropout, max_length):
    source_attention = MultiHeadedAttention(heads, d_model, dropout)
    target_attention = MultiHeadedAttention(heads, d_model, dropout)
    cross_attention = MultiHeadedAttention(heads, d_model, dropout)
    feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
    encoder_layer = EncoderLayer(d_model, source_attention, feed_forward, dropout)
    decoder_layer = DecoderLayer(d_model, target_attention, cross_attention, feed_forward, dropout)
    encoder = Encoder(encoder_layer, layers)
    decoder = Decoder(decoder_layer, layers)
    source_embedding = nn.Sequential(Embeddings(source_vocab_size, d_model), PositionalEncoding(d_model, dropout, max_length))
    target_embedding = nn.Sequential(Embeddings(target_vocab_size, d_model), PositionalEncoding(d_model, dropout, max_length))
    return EncoderDecoder(
        encoder,
        decoder,
        source_embedding,
        target_embedding,
        Generator(d_model, target_vocab_size),
    )


def run_epoch(model, loader, optimizer, criterion, device, target_pad, train):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    for source, target in loader:
        source = source.to(device)
        target = target.to(device)
        decoder_input = target[:, :-1]
        expected = target[:, 1:]
        source_mask = (source != target_pad).unsqueeze(1)
        target_padding_mask = (decoder_input != target_pad).unsqueeze(1)
        target_mask = target_padding_mask & subsequent_mask(decoder_input.size(1), device)

        with torch.set_grad_enabled(train):
            hidden = model(source, decoder_input, source_mask, target_mask)
            log_probabilities = model.generator(hidden)
            loss = criterion(log_probabilities.reshape(-1, log_probabilities.size(-1)), expected.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        token_count = (expected != target_pad).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Train the Transformer translation model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    en_vocab = load_vocab(args.data_dir / "en.json")
    zh_vocab = load_vocab(args.data_dir / "zh.json")
    pairs = read_parallel_data(args.data_dir, en_vocab, zh_vocab, args.max_length)
    dataset = TranslationDataset(pairs)
    validation_size = max(1, int(len(dataset) * args.validation_ratio))
    training_size = len(dataset) - validation_size
    training_set, validation_set = random_split(dataset, [training_size, validation_size])
    collate_fn = make_collate_fn(en_vocab["<PAD>"], zh_vocab["<PAD>"])
    train_loader = DataLoader(dataset=training_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(dataset=validation_set, batch_size=args.batch_size, collate_fn=collate_fn)

    model = build_model(
        len(en_vocab), len(zh_vocab), args.d_model, args.heads, args.layers,
        args.d_ff, args.dropout, args.max_length,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss(ignore_index=zh_vocab["<PAD>"], reduction="sum")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using {device}; {len(training_set)} training pairs, {len(validation_set)} validation pairs")

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint = args.output_dir / "transformer_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, zh_vocab["<PAD>"], True)
        validation_loss = run_epoch(model, validation_loader, optimizer, criterion, device, zh_vocab["<PAD>"], False)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "en_vocab": en_vocab,
                    "zh_vocab": zh_vocab,
                },
                best_checkpoint,
            )
            saved = f"saved best {best_checkpoint}"
        else:
            epochs_without_improvement += 1
            saved = f"no improvement ({epochs_without_improvement}/{args.patience})"

        print(f"epoch {epoch:02d} | train loss {train_loss:.4f} | validation loss {validation_loss:.4f} | {saved}")
        if epochs_without_improvement >= args.patience:
            print(f"early stopping after {args.patience} epochs without validation improvement")
            break


if __name__ == "__main__":
    main()