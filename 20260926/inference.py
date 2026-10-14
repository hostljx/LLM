import argparse
from pathlib import Path

import torch

from train import build_model, tokenize_english


DEFAULT_CHECKPOINT = r"D:\PythonProjects\LLM\20260926\checkpoints\transformer_best.pt"


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def subsequent_mask(size, device):
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))


def translate(model, text, en_vocab, zh_vocab, device, max_length):
    source_tokens = tokenize_english(text)
    source_ids = [en_vocab["<SOS>"]]
    source_ids.extend(en_vocab.get(token, en_vocab["<UNK>"]) for token in source_tokens)
    source_ids.append(en_vocab["<EOS>"])
    if len(source_ids) > max_length:
        raise ValueError(f"Input is too long: {len(source_ids)} tokens, maximum is {max_length}")

    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    source_mask = (source != en_vocab["<PAD>"]).unsqueeze(1)
    target_ids = [zh_vocab["<SOS>"]]
    target_tokens = {index: token for token, index in zh_vocab.items()}

    with torch.inference_mode():
        memory = model.encode(source, source_mask)
        for _ in range(max_length - 1):
            target = torch.tensor([target_ids], dtype=torch.long, device=device)
            target_mask = subsequent_mask(target.size(1), device).unsqueeze(0)
            hidden = model.decode(memory, source_mask, target, target_mask)
            next_id = model.generator(hidden[:, -1]).argmax(dim=-1).item()
            target_ids.append(next_id)
            if next_id == zh_vocab["<EOS>"]:
                break

    return "".join(
        target_tokens.get(index, "<UNK>")
        for index in target_ids[1:]
        if index not in {zh_vocab["<EOS>"], zh_vocab.get("<PAD>", -1)}
    )


def main():
    parser = argparse.ArgumentParser(description="Translate English text with the trained Transformer.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--text", help="English text to translate. If omitted, read from stdin.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    saved_args = checkpoint["args"]
    en_vocab = checkpoint["en_vocab"]
    zh_vocab = checkpoint["zh_vocab"]
    model = build_model(
        len(en_vocab),
        len(zh_vocab),
        saved_args["d_model"],
        saved_args["heads"],
        saved_args["layers"],
        saved_args["d_ff"],
        saved_args["dropout"],
        saved_args["max_length"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.text is not None:
        print(translate(model, args.text, en_vocab, zh_vocab, device, saved_args["max_length"]))
        return

    while True:
        try:
            text = input("English (blank to exit): ")
        except EOFError:
            print()
            break

        if not text.strip():
            break

        try:
            print(translate(model, text, en_vocab, zh_vocab, device, saved_args["max_length"]))
        except ValueError as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
