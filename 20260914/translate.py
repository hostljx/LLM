import torch
import cfg
from model import Encoder, Decoder
from data import Vocabulary, zh_tokenize, en_tokenize


'''(cls, file_name, json_key)'''
zh_vocab = Vocabulary.load(
    "word2index.json", "word2index"
)

en_vocab = Vocabulary.load(
    "en_word2index.json", "en_word2index"
)


def encoder_zh(text):
    """将中文文本编码为索引序列"""
    tokens = list(zh_tokenize(text))
    return [
        zh_vocab.word2index.get(token, zh_vocab.UNK_INDEX)
        for token in tokens
    ]

def decoder_en(ids):
    """将索引序列解码为英文文本"""
    words = []

    for index in ids:
        if index == en_vocab.EOS_INDEX:
            break
        if index not in (en_vocab.PAD_INDEX, en_vocab.SOS_INDEX):
            words.append(en_vocab.index2word.get(index, en_vocab.UNK_TOKEN))

    return " ".join(words)

def translate(text, encoder, decoder, device):
    source_ids = encoder_zh(text)
    source = torch.tensor(
        source_ids,
        dtype = torch.long,
        device = device
    ).unsqueeze(0)  # 添加batch维度

    '''中文编码'''
    encoder_outputs, hidden = encoder(source)

    '''解码器的初始输入为SOS_TOKEN'''
    decoder_input = torch.tensor(
        [[en_vocab.SOS_INDEX]],
        dtype = torch.long,
        device = device
    )

    result_ids = []

    for _ in range(100):  # 最大解码长度为100
        output, hidden = decoder(
            decoder_input, 
            hidden, 
            encoder_outputs
        )

        next_id = output.argmax(dim=-1)[:, -1]
        next_id_value = next_id.item()

        if next_id_value == en_vocab.EOS_INDEX:
            break

        result_ids.append(next_id_value)
        decoder_input = next_id.unsqueeze(1)  # 为下一步解码准备输入

    return decoder_en(result_ids)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(len(zh_vocab.vocab)).to(device)
    decoder = Decoder(len(en_vocab.vocab)).to(device)

    checkpoint = torch.load(
        cfg.path / "model" / "translation_model.pth",
        map_location=device
    )

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    while True:
        text = input("中文：").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break

        with torch.no_grad():
            result = translate(text, encoder, decoder, device)

        print("英文：", result)

