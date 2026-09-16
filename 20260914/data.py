

"""
    数据模块

    转简体中文：
        pip install zhconv

"""
from cfg import Config
import cfg
import jieba
import nltk
import json
import zhconv
from collections import Counter


'''创建nltk分词器对象'''
__en_tokenizer = nltk.TreebankWordTokenizer()


def load_train(data_path = Config.FILE_PATH):
    """加载语料库

    Args:
        data_path (train.txt): path

    Returns:
        iterator: 逐行生成的句对迭代器
    """    
    with open(data_path, 'rt', encoding='utf-8') as f:
        # for line in f:
        #             words = line.split("\t")
        #             en = words[0]
        #             zh = words[1]

        '''加val'''
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue

            words = line.split("\t", 1)
            if len(words) != 2:
                raise ValueError(
                    f"Invalid translation pair in {data_path}, "
                    f"line {line_number}: expected a tab separator"
                )

            en, zh = words
            yield zh, en


def zh_tokenize(zh_text):
    """中文分词

    Args:
        zh_text (): 待分词的中文文本

    Yields:
        _type_: 分词结果
    """    
    tokens = jieba.lcut(zhconv.convert(zh_text, 'zh-cn'))
    yield from tokens

def en_tokenize(en_text):
    """英文分词

    Args:
        en_text (): 待分词的英文文本

    Yields:
        _type_: 分词结果
    """
    yield from __en_tokenizer.tokenize(en_text)


class Vocabulary:
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
    PAD_INDEX, SOS_INDEX, EOS_INDEX, UNK_INDEX = 0, 1, 2, 3
    __INIT_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]



    def __init__(self):
        self.vocab = list(self.__INIT_TOKENS)
        self.word2index = {}
        self.index2word = {}
        self.counter = Counter()

    def generate_vocab(self, text_type, tokenize_func):
        '''type = "zh" or "en"'''
        if text_type not in ("zh", "en"):
            raise ValueError("text_type must be 'zh' or 'en'")

        for zh_text, en_text in load_train():
            if text_type == "zh":
                tokens = tokenize_func(zh_text)
            else:
                tokens = tokenize_func(en_text)

            self.counter.update(tokens)

        '''构建词汇表'''
        self.vocab += [token for token in self.counter] 
        # print(self.vocab)


        '''构建word2index和index2word'''
        self.word2index = {word: index for index, word in enumerate(self.vocab)}
        self.index2word = {index: word for index, word in enumerate(self.vocab)}


    def encode(self, text, tokenize_func):
        """将文本编码为索引序列"""
        words = list(tokenize_func(text))
        return [self.word2index.get(word, self.UNK_INDEX) for word in words]

    def decode(self, ids):
        """将索引序列解码回文本"""
        return [self.index2word.get(index, self.UNK_TOKEN) for index in ids]
        

    def save(self):
        # 将word2index保存到文件中
        with open(cfg.path / "en_word2index.json", "wt", encoding="utf-8") as f:
            json.dump({"en_word2index": self.word2index}, f, ensure_ascii=False, indent=2)

    
    @classmethod
    def load(cls, file_name, json_key):
        vocab = cls()
        # 从文件中加载word2index
        with open(cfg.path / file_name, 'rt', encoding="utf-8") as f:
            vocab.word2index = json.load(f)[json_key]

        vocab.index2word = {int(index): word for word, index in vocab.word2index.items()}
        '''sorted会按数字索引排序'''
        vocab.vocab = [word for index, word in sorted(vocab.index2word.items())]

        return vocab
        


if __name__ == "__main__":
    # res = en_tokenize("I am a student.")
    # print(res)

    vocab = Vocabulary()
    vocab.generate_vocab("en", en_tokenize)
    vocab.save()
    # vocab.load("word2index.json", "word2index")
    # ids1 = vocab.encode("我爱海洋", zh_tokenize)
    # print(ids1)
    ids2 = vocab.encode("我是志愿者", zh_tokenize)
    print(ids2)

    wds = vocab.decode(ids2)
    print(wds)
