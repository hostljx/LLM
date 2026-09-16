from gensim.models import KeyedVectors


mv = KeyedVectors.load_word2vec_format(r'20260909\datas\sgns.weibo.word.bz2')

print(mv)
print(mv['中国'])

print(mv.most_similar('成都', topn=6))

#   词转为向量 表达
print(mv.most_similar(positive=['男人', '女生'], negative=['男生'], topn=1))