import os
import collections
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F

jieba.add_word('大模型')

with open("20260904/data/info.txt", "r", encoding="utf-8") as f:
    sentences = [jieba.lcut(line.strip()) for line in f.readlines()]
print("分词结果：")
for s in sentences:
    print(s)

vocab = ['PAD']
counter = collections.Counter(w for sent in sentences for w in sent)
for w in counter:
    vocab.append(w)

word2index = {w: i for i, w in enumerate(vocab)}
index2word = {i: w for i, w in enumerate(vocab)}

VOCAB_SIZE = len(vocab)

print(f"\n== 词表 (共 {VOCAB_SIZE} 词, 含 PAD) ==")
print(vocab)

WINDOW = 2 # 中心词左右各看2个词
EMBED_DIM = 16 # 词向量维度
EPOCHS = 300
LR = 0.05

def make_samples(window=WINDOW):
    
    X, y = [], []
    for sent in sentences:
        n = len(sent)
        for i in range(n):
            ctx = []
            for j in range(i - window, i + window + 1):
                if j == i: # 跳过中心词自己
                    continue
                if 0 <= j < n:
                    ctx.append(sent[j])
                else: # 出界 -> PAD
                    ctx.append('PAD')
            X.append([word2index[w] for w in ctx])
            y.append(word2index[sent[i]])
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

X, y = make_samples()
print(f"\n== 训练样本 (共 {len(X)} 个) ==")
for i in range(min(5, len(X))):
    ctx_words = [index2word[idx] for idx in X[i].tolist()]
    print(f"上下文 {ctx_words} -> 预测中心词 [{index2word[y[i].item()]}]")


# ------------------- 4. 定义 CBOW 模型 -------------------
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # W1: Embedding 层, 每行就是一个词的向量 (训练后取它的weight就是词向量)
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        # W2: 隐层 -> 词表分数
        self.linear = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        emb = self.embeddings(x) # (batch, ctx_len, embed_dim) 查表得到上下文各词向量
        h = emb.mean(dim=1) # (batch, embed_dim) 取平均 = 词袋
        logits = self.linear(h) # (batch, vocab_size) 每个候选词的分数
        return logits

model = CBOW(VOCAB_SIZE, EMBED_DIM)
loss_fn = nn.CrossEntropyLoss() # 内部自带 softmax, 所以模型输出不用再过softmax
optimizer = torch.optim.SGD(model.parameters(), lr=LR)


# 训练
print("== 训练 ==")
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch {epoch:>4d} | loss = {loss.item():.4f}")

# 验证: 词向量相似度
def most_similar(word, k=3):
    """找与word最相似的k个词"""
    wv = model.embeddings.weight.data # (vocab_size, embed_dim)
    v = wv[word2index[word]]
    sims = F.cosine_similarity(v.unsqueeze(0), wv) # word与词表每个词的余弦
    top = sims.argsort(descending=True)
    return [(index2word[i.item()], round(sims[i].item(), 3))
        for i in top if i != word2index[word]][:k]

print("== 词向量相似度(训练后) ==")
for w in ["我", "喜欢", "的", "课", "大模型", "开心"]:
    if w in word2index:
        print(f"{w} 的近义词: {most_similar(w)}")


# 验证
def predict(ctx_words, k=3):
    ctx_idx = torch.tensor([[word2index[w] for w in ctx_words]])
    probs = F.softmax(model(ctx_idx), dim=-1)[0]
    top = probs.argsort(descending=True)
    return [(index2word[i.item()], round(probs[i.item()].item(), 3)) for i in top[:k]]
print("\n== 上下文预测演示 ==")
print(f"上下文 ['我','喜欢','的','课'] 最可能是: {predict(['我', '喜欢', '的', '课'])}")
print(f"上下文 ['PAD','我','大模型','的'] 最可能是: {predict(['PAD', '我', '大模型', '的'])}")
print(f"上下文 ['喜欢','大模型','课','PAD'] 最可能是: {predict(['喜欢', '大模型', '课', 'PAD'])}")

# ------------------- 8. 词向量可视化 (PCA 降到 2D) -------------------
# 语义相似的词, 词向量方向接近 -> 在图上聚在一起
import numpy as np
import matplotlib
matplotlib.use("Agg") # 不弹窗, 直接存图
import matplotlib.pyplot as plt

# 中文显示 (Windows)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
wv = model.embeddings.weight.data.numpy() # (vocab_size, embed_dim), 含 PAD
wv_center = wv - wv.mean(axis=0, keepdims=True)
# 手写 PCA: 对协方差做 SVD, 取前两个主成分方向投影
_, _, Vt = np.linalg.svd(wv_center, full_matrices=False)
xy = wv_center @ Vt[:2].T # 每个词 -> 2D 坐标

fig, ax = plt.subplots(figsize=(9, 7))
for i, w in enumerate(vocab):
    if w == 'PAD':
        continue
    ax.scatter(xy[i, 0], xy[i, 1], s=120)
    ax.annotate(w, (xy[i, 0], xy[i, 1]), fontsize=14,
        xytext=(6, 6), textcoords="offset points")

ax.set_title("CBOW 词向量 (PCA 降到 2D)")
ax.grid(alpha=0.3)
fig_path = os.path.join("20260904/data", "word_vectors_pca.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n== 词向量图已保存: {fig_path} ==")