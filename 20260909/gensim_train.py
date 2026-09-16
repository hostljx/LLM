from gensim.models import Word2Vec


sentences = [["成都", "麻将", "天府", "芙蓉"]]

model = Word2Vec(sentences, vector_size=128, window=2, min_count=1, sg=0)   # sg =0/ 1 0: CBOW, 1: Skip-gram

print(model.wv.index_to_key)  # 获取词向量
print(model.wv['成都'])

# 保存
model.wv.save_word2vec_format(r'20260909/gensim.txt')


