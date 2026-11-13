#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_rag.py —— 入库结果验证
=====================================================================
验证 pdf_ingest.py 到底入了个什么东西、检索能不能用。
不写库、不删数据，纯只读体检。

用法：
    python verify_rag.py                          # 用默认 collection 全量体检
    python verify_rag.py --collection pdf_rag_47
    python verify_rag.py --query "第三章讲的啥" --k 5     # 顺带看检索结果
    python verify_rag.py --query "怎么配置" --ask        # 再用 Ollama 生成答案（端到端）
    python verify_rag.py --src 手册.pdf --query "xxx"    # 只在该文件内检索
    python verify_rag.py --llm qwen3:8b --ask
"""

import argparse
import json
import re
import sys
import urllib.request

try:
    if (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# =====================================================================
# 配置（跟 pdf_ingest.py 保持一致）
# =====================================================================
import os

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "pdf_rag_47")

CHECKS = []          # [(是否通过, 检查项, 说明)]


def check(ok, name, detail=""):
    CHECKS.append((bool(ok), name, detail))
    flag = "✅" if ok else "❌"
    print(f"  {flag} {name}" + (f" —— {detail}" if detail else ""))
    return ok


def hr(title=""):
    print("\n" + "=" * 68)
    if title:
        print(title)
        print("=" * 68)


def _dtype(ftype):
    """pymilvus 2.6 的 f['type'] 是 int 枚举，转成 'VarChar' 这种可读名"""
    try:
        from pymilvus import DataType
        return DataType(ftype).name
    except Exception:
        s = str(ftype)
        return s.split(".")[-1]


META_KEYS = ("source", "page", "header_path", "h1", "h2", "h3", "chunk_index", "char_len")


def _meta_of(row):
    """元数据既可能是独立字段（本脚本的 schema），也可能是 metadata JSON 串，两种都兜"""
    meta = {k: row[k] for k in META_KEYS if k in row}
    raw = row.get("metadata")
    if isinstance(raw, dict):
        for k, v in raw.items():
            meta.setdefault(k, v)
    elif isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            for k, v in json.loads(raw).items():
                meta.setdefault(k, v)
        except Exception:
            pass
    return meta


# =====================================================================
# 1. 连 Milvus，拿 collection 概况
# =====================================================================
def inspect_collection(uri, coll):
    from pymilvus import MilvusClient

    hr("[1] Milvus 连接与 collection 概况")
    try:
        client = MilvusClient(uri=uri)
        try:
            version = client.get_server_version()
        except Exception:
            version = "?"
        check(True, "Milvus 可连接", f"{uri} (server {version})")
    except Exception as e:
        check(False, "Milvus 可连接", f"{uri} -> {e}")
        return None, None

    exists = client.has_collection(coll)
    if not check(exists, f"collection「{coll}」存在"):
        cols = client.list_collections()
        print(f"      当前库里的 collection：{cols}")
        return client, None

    # 新写入的数据要 flush 才计入统计
    try:
        client.flush(coll)
    except Exception:
        pass
    stats = client.get_collection_stats(coll)
    rows = int(stats.get("row_count", 0))
    if rows == 0:      # 统计偶发滞后，用 count(*) 兜一次
        try:
            res = client.query(coll, output_fields=["count(*)"])
            rows = int(res[0].get("count(*)", 0)) if res else 0
        except Exception:
            pass
    check(rows > 0, "实体数 > 0", f"{rows} 条")
    print(f"      实体总数：{rows}")

    desc = client.describe_collection(coll)
    print("\n      字段清单：")
    pk_name, text_field = None, None
    for f in desc["fields"]:
        fname, ftype = f["name"], _dtype(f["type"])
        extra = ""
        if f.get("is_primary"):
            pk_name, extra = fname, " [primary]"
        if "dense" in fname or "sparse" in fname:
            extra += " [vector]"
        if fname in ("text", "page_content") and "varchar" in ftype.lower():
            text_field = fname
        print(f"        - {fname:<14} {ftype:<12}{extra}")

    check(pk_name is not None, "找到主键字段", pk_name or "缺失")
    check(text_field is not None, "找到正文字段", text_field or "缺失")

    # 度量类型
    try:
        for idx_name in ("dense", "sparse"):
            try:
                info = client.describe_index(coll, index_name=idx_name)
                metric = info.get("metric_type")
                if metric:
                    print(f"      索引 {idx_name} 度量类型：{metric}")
            except Exception:
                pass
    except Exception:
        pass

    return client, text_field


# =====================================================================
# 2. 抽样看内容质量
# =====================================================================
def sample_rows(client, coll, text_field, n=5):
    hr(f"[2] 内容抽样（{n} 条）")
    desc = client.describe_collection(coll)
    pk = next(f["name"] for f in desc["fields"] if f.get("is_primary"))
    pk_type = _dtype(next(f["type"] for f in desc["fields"] if f.get("is_primary")))
    expr = f'{pk} != ""' if "varchar" in pk_type.lower() else f"{pk} >= 0"

    try:
        rows = client.query(
            collection_name=coll, filter=expr,
            output_fields=["*"], limit=n,
        )
    except Exception as e:
        check(False, "查询抽样", str(e))
        return []

    if not rows:
        check(False, "查询抽样", "查不到行")
        return []

    ok = 0
    for i, r in enumerate(rows, 1):
        body = (r.get(text_field) or "").strip()
        meta = _meta_of(r)
        if len(re.sub(r"\s", "", body)) >= 20:
            ok += 1
        print(f"\n  --- 样本 {i} | source={meta.get('source')} "
              f"| page={meta.get('page')} | len={len(body)}")
        print(f"      标题路径：{meta.get('header_path') or '(无)'}")
        print("      " + body[:260].replace("\n", "\n      "))

    check(ok == len(rows), f"抽样 {len(rows)} 条正文非空且长度正常", f"合格 {ok}/{len(rows)}")
    return rows


# =====================================================================
# 3. 分布体检
# =====================================================================
def distribution(client, coll, text_field, limit=500):
    hr("[3] 切片分布体检")
    desc = client.describe_collection(coll)
    pk = next(f["name"] for f in desc["fields"] if f.get("is_primary"))
    pk_type = _dtype(next(f["type"] for f in desc["fields"] if f.get("is_primary")))
    expr = f'{pk} != ""' if "varchar" in pk_type.lower() else f"{pk} >= 0"

    rows = client.query(collection_name=coll, filter=expr,
                        output_fields=["*"], limit=limit)
    if not rows:
        check(False, "分布采样", "查不到行")
        return
    texts = [(r.get(text_field) or "") for r in rows]
    lens = sorted(len(t) for t in texts)

    src_counter = {}
    page_missing = 0
    for r in rows:
        meta = _meta_of(r)
        src_counter[meta.get("source", "?")] = src_counter.get(meta.get("source", "?"), 0) + 1
        if meta.get("page", -1) in (-1, None, ""):
            page_missing += 1

    print(f"      抽样 {len(texts)} 条（总实体数可能更多）")
    print(f"      长度 min={lens[0]}  p50={lens[len(lens)//2]}  "
          f"p90={lens[int(len(lens)*0.9)]}  max={lens[-1]}")
    print(f"      各文件条数：{src_counter}")

    very_short = sum(1 for t in texts if len(re.sub(r"\s", "", t)) < 30)
    very_long = sum(1 for t in texts if len(t) > 2000)
    dup = len(texts) - len(set(texts))

    check(very_short <= max(2, len(texts) // 50), "极少碎片块（<30字）", f"{very_short} 条")
    check(very_long <= max(2, len(texts) // 20), "极少超长块（>2000字）", f"{very_long} 条")
    check(dup <= max(1, len(texts) // 100), "重复块可控", f"{dup} 条完全重复")
    if page_missing:
        print(f"      [!] {page_missing} 条 page=-1（该文档没有 content_list.json，不影响检索）")
    else:
        check(True, "页码映射完整")


# =====================================================================
# 4. 嵌入服务 + 检索
# =====================================================================
def build_store(coll):
    """
    同 pdf_ingest.get_store：修 langchain-milvus 0.3.x + pymilvus 2.6 的别名 bug，
    否则集合已存在时构造就抛 ConnectionNotExistException。
    """
    from pymilvus import MilvusClient, connections

    alias = "rag_shared"
    if alias not in [a for a, _ in connections.list_connections()]:
        connections.connect(alias=alias, uri=MILVUS_URI)

    _orig_init = MilvusClient.__init__

    def _patched_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        self._using = alias

    MilvusClient.__init__ = _patched_init

    from langchain_milvus import BM25BuiltInFunction, Milvus
    from langchain_ollama import OllamaEmbeddings

    embedding = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    store = Milvus(
        embedding_function=embedding,
        connection_args={"uri": MILVUS_URI},
        builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
        vector_field=["dense", "sparse"],
        collection_name=coll,
    )
    return store, embedding


def verify_embedding(embedding):
    hr("[4] 嵌入模型")
    try:
        v = embedding.embed_query("连通性测试")
        check(True, "Ollama embedding 可用", f"{EMBED_MODEL} 维度={len(v)}")
        return len(v)
    except Exception as e:
        check(False, "Ollama embedding 可用", f"{OLLAMA_BASE_URL} / {EMBED_MODEL} -> {e}")
        return None


def retrieval_demo(store, query, k, expr=None):
    hr(f"[5] 检索验证：{query!r}")

    def show(tag, docs, scores=None):
        print(f"\n  【{tag}】")
        if not docs:
            print("    （空）")
            return 0
        for i, d in enumerate(docs, 1):
            meta = d.metadata or {}
            sc = f" score={scores[i-1]:.4f}" if scores else ""
            print(f"    {i}. [{meta.get('source')} p{meta.get('page')}] "
                  f"{meta.get('header_path') or '(无标题)'}{sc}")
            print("       " + d.page_content[:130].replace("\n", " "))
        return len(docs)

    # 稠密单路
    try:
        kw = {"k": k}
        if expr:
            kw["expr"] = expr
        pairs = store.similarity_search_with_score(query, **kw)
        dense_docs = [p[0] for p in pairs]
        dense_scores = [p[1] for p in pairs]
        nd = show("稠密向量检索", dense_docs, dense_scores)
    except Exception as e:
        check(False, "稠密检索", str(e))
        dense_docs, nd = [], 0
    else:
        check(nd > 0, "稠密检索返回结果", f"{nd} 条")

    # 混合 RRF
    try:
        kwargs = {"k": k, "fetch_k": max(k * 4, 20), "rank_type": "rrf", "ranker_params": {"k": 60}}
        if expr:
            kwargs["expr"] = expr
        retriever = store.as_retriever(search_kwargs=kwargs)
        hybrid_docs = retriever.invoke(query)
        nh = show("混合检索（dense + BM25，RRF 融合）", hybrid_docs)
    except Exception as e:
        check(False, "混合检索（RRF）", str(e))
        hybrid_docs, nh = [], 0
    else:
        check(nh > 0, "混合检索返回结果", f"{nh} 条")

    # 两路重合度，判断 BM25 到底有没有起作用
    if nd and nh:
        # 用内容指纹比，doc 对象不是同一个
        dkeys = {d.page_content[:60] for d in dense_docs}
        hkeys = {d.page_content[:60] for d in hybrid_docs}
        overlap = len(dkeys & hkeys)
        print(f"\n  两路重合 {overlap}/{k} 条"
              + ("" if overlap < k else " —— 完全重合，BM25 可能没生效（检查 sparse 字段/中文分词）"))

    return hybrid_docs or dense_docs


# =====================================================================
# 5. 端到端问答
# =====================================================================
def pick_llm(want=None):
    if want:
        return want
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
    except Exception:
        return None
    # embedding / rerank 模型不支持 chat，必须排掉
    bad = ("embed", "bge", "gte", "e5", "rerank", "minilm")
    names = [n for n in names if not any(k in n.lower() for k in bad)]
    for kw in ("qwen3", "qwen2.5", "qwen", "deepseek", "llama3", "llama"):
        for n in names:
            if n.startswith(kw):
                return n
    return names[0] if names else None


def ask_llm(docs, query, model, num_gpu=None):
    hr(f"[6] 端到端问答（模型：{model}{'，强制 CPU' if num_gpu == 0 else ''}）")
    from langchain_ollama import ChatOllama

    ctx_parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        ctx_parts.append(f"[{i}] （{m.get('source')} 第{m.get('page')}页 "
                         f"{m.get('header_path') or ''}）\n{d.page_content}")
    ctx = "\n\n".join(ctx_parts)

    prompt = (
        "你是文档问答助手。只能依据下面的资料回答，不要编造。\n"
        "如果资料里没有答案，直接说“资料中没有找到相关内容”。\n"
        "回答时用 [1][2] 标注引用编号。\n\n"
        f"=== 资料 ===\n{ctx}\n\n=== 问题 ===\n{query}\n\n=== 回答 ==="
    )
    kwargs = {"num_ctx": 8192}
    if num_gpu is not None:
        kwargs["num_gpu"] = num_gpu
    try:
        # reasoning=False -> 关掉思考模式。qwen3.5 这类思考模型在 4096 上下文里
        # 会把 token 全烧在推理上，正文被截断，表现就是"返回 0 字"
        llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0.1,
                         reasoning=False, **kwargs)
    except TypeError:      # 老版本 langchain_ollama 没有 reasoning 参数
        llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0.1, **kwargs)
    try:
        ans = llm.invoke(prompt).content
    except Exception as e:
        msg = str(e)
        check(False, "LLM 生成", msg)
        if "CUDA error" in msg or "shared object" in msg or "0xc0000409" in msg:
            print("\n  [!] 这是 Ollama 的 CUDA 后端加载不了内核，不是检索链路的问题。两条路：")
            print("      1) 让 Ollama 用 EAGER 方式加载 CUDA 模块（推荐，保留 GPU 速度）：")
            print("         setx CUDA_MODULE_LOADING EAGER")
            print("         然后退出托盘里的 Ollama（右键 Quit），再重新打开，重跑本命令")
            print("      2) 强制 CPU 出答案（慢，但一定能跑通）：")
            print("         verify_rag.py --ask --cpu")
        return
    if not ans.strip():
        check(False, "LLM 生成成功", "返回空内容（多半是思考模式把上下文烧光了）")
        return
    check(bool(ans.strip()), "LLM 生成成功", f"{len(ans)} 字")
    print("\n  答案：\n" + "\n".join("    " + ln for ln in ans.split("\n")))
    print("\n  引用来源：")
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        print(f"    [{i}] {m.get('source')} p{m.get('page')} / {m.get('header_path') or '(无标题)'}")


# =====================================================================
# main
# =====================================================================
def main():
    ap = argparse.ArgumentParser("验证 pdf -> milvus 入库结果")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--query", default="", help="验证用查询语句；不给就只做体检")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--src", default="", help="只在某个 source 文件内检索")
    ap.add_argument("--ask", action="store_true", help="调 Ollama 生成答案（端到端）")
    ap.add_argument("--llm", default="", help="指定 Ollama 模型名，默认自动挑")
    ap.add_argument("--cpu", action="store_true", help="LLM 强制走 CPU（GPU 报 CUDA 错误时兜底）")
    ap.add_argument("--sample", type=int, default=5)
    args = ap.parse_args()

    print(f"目标：{MILVUS_URI} / collection = {args.collection}")

    client, text_field = inspect_collection(MILVUS_URI, args.collection)
    if client is None:
        return summary()
    if text_field is None:
        return summary()

    sample_rows(client, args.collection, text_field, args.sample)
    distribution(client, args.collection, text_field)

    store, embedding = build_store(args.collection)
    dim = verify_embedding(embedding)

    docs = []
    if args.query:
        expr = None
        if args.src:
            # 本脚本的 schema 把 source 存成独立字段；兼容只存 metadata 的老库
            fields = {f["name"] for f in client.describe_collection(args.collection)["fields"]}
            key = "source" if "source" in fields else 'metadata["source"]'
            expr = f'{key} == {json.dumps(args.src, ensure_ascii=False)}'
        docs = retrieval_demo(store, args.query, args.k, expr=expr)
        if args.ask and docs:
            model = pick_llm(args.llm or None)
            if not model:
                check(False, "LLM 可用", f"{OLLAMA_BASE_URL} 上没有可用模型")
            else:
                ask_llm(docs, args.query, model, num_gpu=0 if args.cpu else None)
    else:
        print("\n  （没给 --query，跳过检索验证。想看检索：--query \"你的问题\"）")

    summary()


def summary():
    hr("体检汇总")
    fail = [c for c in CHECKS if not c[0]]
    print(f"  通过 {len(CHECKS) - len(fail)}/{len(CHECKS)} 项")
    if fail:
        print("\n  需要处理：")
        for _, name, detail in fail:
            print(f"    ❌ {name} —— {detail}")
    else:
        print("  ✅ 全部通过，链路可用")
    hr()
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
