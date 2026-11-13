"""
pdf_ingest.py —— PDF 全链路入库（MinerU 4.x）
=====================================================================
PDF ──mineru-kit parse──> markdown.md ──清洗──> 切片 ──> Milvus(稠密+BM25稀疏) 入库

依赖：
    pip install "mineru>=4.0,<5"
    pip install langchain-milvus langchain-ollama langchain-text-splitters pymilvus tqdm

MinerU 4.x 产物（本脚本用 --format zip 一次拿全，解包到输出目录）：
    markdown.md           正文
    images/               图片
    middle_json.json      带页码的中间结构（本脚本用它反查页码）
    structured_content.json

用法：
    python pdf_ingest.py 手册.pdf                   # 解析 + 清洗 + 切片 + 入库
    python pdf_ingest.py 手册.pdf --pages 1-50      # 只处理前 50 页（大书先试水）
    python pdf_ingest.py 手册.pdf --dry-run         # 只解析切片打印样例，不碰数据库
    python pdf_ingest.py 手册.pdf --md mineru_out/手册/markdown.md   # 跳过解析
    python pdf_ingest.py 手册.pdf --tier basic      # 慢的话降档
    python pdf_ingest.py 手册.pdf --recreate --yes  # 删掉整个 collection 重建（破坏性）
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Windows 终端中文输出兜底
try:
    if (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from tqdm import tqdm
except ImportError:  # 没装 tqdm 就静默降级
    def tqdm(iterable, **kwargs):
        return iterable


# =====================================================================
# 一、配置区
# =====================================================================
MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")

DEFAULT_COLLECTION = os.getenv("MILVUS_COLLECTION", "pdf_rag_47")
DEFAULT_CHUNK_SIZE = 800      # 中文字符数，约 500~800 汉字一个语义块
DEFAULT_OVERLAP = 120         # 重叠，防止切在句子中间丢上下文
MIN_CHUNK_CHARS = 20          # 去空白后少于这么多字的块直接丢（公式碎片、残行）
BATCH_SIZE = 64               # 每批入库条数

MINERU_OUT_ROOT = Path(os.getenv("MINERU_OUT_ROOT", "mineru_out"))

# 本地已装好的 MinerU 4.x（独立 uv 环境，mineru.cmd 里 MINERU_HOME 指向 .mineru）
MINERU_KIT = os.getenv(
    "MINERU_KIT",
    r"D:\PythonProjectAll\LLM\minerU\.mineru\Scripts\mineru-kit.exe",
)


# =====================================================================
# 二、MinerU 4.x 调用
# =====================================================================
def find_mineru_kit(override=""):
    """优先级：命令行指定 > 配置里的 MINERU_KIT > 当前 Python 同目录 Scripts > PATH"""
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise SystemExit(f"--mineru-cmd 指向的文件不存在：{override}")
        return str(p)

    if MINERU_KIT and Path(MINERU_KIT).is_file():
        return MINERU_KIT

    py_dir = Path(sys.executable).parent
    for cand in (py_dir / "mineru-kit.exe", py_dir / "mineru-kit"):
        if cand.is_file():
            return str(cand)
    found = shutil.which("mineru-kit")
    if found:
        return found

    print("[X] 没找到 mineru-kit。装一个：")
    print('    pip install "mineru>=4.0,<5" -i https://mirrors.aliyun.com/pypi/simple/')
    print("    或者用 --mineru-cmd 指向现成的 mineru-kit.exe")
    raise SystemExit(1)


def _mineru_home(exe_path: str):
    """从 .../.mineru/Scripts/mineru-kit.exe 反推 MINERU_HOME，让模型走本地那份"""
    p = Path(exe_path).resolve()
    if p.parent.name.lower() == "scripts":
        home = p.parent.parent
        if (home / "config.yaml").is_file() or (home / "models").is_dir():
            return home
    return None


def run_mineru(pdf_path: Path, out_root: Path, tier: str, mineru_cmd="", pages=""):
    """返回 (md 路径, 输出目录)。pages 形如 '1-50'，空表示全本。"""
    out_dir = out_root / (pdf_path.stem + (f"_p{pages.replace(',', '_')}" if pages else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = find_mineru_kit(mineru_cmd)
    # -f zip 一次拿到 markdown.md + middle_json.json（-f markdown 不含页码信息）
    zip_path = out_dir / "result.zip"
    cmd = [exe, "parse", str(pdf_path), "-o", str(zip_path), "--tier", tier, "--format", "zip"]
    if pages:
        cmd += ["--pages", pages]

    print("[1/5] MinerU 解析中…（首次运行要下模型，几分钟正常）")
    print(f"      {' '.join(cmd)}")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MINERU_TELEMETRY_DISABLED", "1")
    home = _mineru_home(exe)
    if home:
        env["MINERU_HOME"] = str(home)
        print(f"      MINERU_HOME={home}（模型走本地这份，不会重新下载）")
    proc = subprocess.run(cmd, capture_output=False, text=True,
                          encoding="utf-8", errors="replace", env=env)

    if proc.returncode != 0:
        print("[-] MinerU 失败，日志末尾：")
        print(((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()[-2000:])
        raise SystemExit(1)

    # 解包 result.zip -> out_dir（markdown.md / middle_json.json / images/）
    import zipfile
    zips = [zip_path] if zip_path.is_file() else list(out_dir.rglob("*.zip"))
    if not zips:
        raise SystemExit(f"MinerU 跑完了但没生成 zip，去 {out_dir} 看一眼")
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(out_dir)

    md = find_markdown(out_dir)
    if md is None:
        raise SystemExit(f"MinerU 跑完了但 {out_dir} 下没有 .md，去目录里翻一下")
    print(f"[2/5] Markdown 就位：{md}")
    return md, out_dir


def find_markdown(out_dir: Path):
    cands = list(out_dir.rglob("*.md"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


# =====================================================================
# 三、清洗（决定检索质量，比切片参数重要）
# =====================================================================
RE_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
RE_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]{0,80}>")
RE_IMG = re.compile(r"!\[([^\]]{0,120})\]\([^)]{0,300}\)")
RE_PAGE_NO = re.compile(r"^[\s\-–—_·]*(?:第\s*)?\d{1,4}\s*(?:页|/\s*\d{1,4})?[\s\-–—_·]*$")
RE_TOC = re.compile(r"^.{0,80}(?:\.{3,}|…{2,}|·{3,})\s*\d{1,4}\s*$")   # 目录行 xxx.....12
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_BLANK = re.compile(r"\n{3,}")
RE_MULTI_SPACE = re.compile(r"[ \t\u00a0\u3000]{2,}")
RE_MD_MARK = re.compile(r"[#*`>|_~\[\]()\-–—…·]")
RE_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
RE_TOC_TITLE = re.compile(r"(目\s*录|contents|table of contents)", re.IGNORECASE)


def _norm(s: str):
    """去掉空白和 markdown 符号只留文字 —— 页码匹配用（否则 '# 标题' 对不上 '标题'）"""
    return RE_MD_MARK.sub("", re.sub(r"\s", "", s))


def _drop_toc_section(lines):
    """
    整块删掉"目录"章节：命中 "## 目录 / Contents" 后，一路删到下一个同级或更高级标题。
    PDF 的目录页会被 MinerU 转成一堆 "- 2.8 奇异值分解 39"，这类条目进了向量库就是纯噪声。
    """
    out, skipping, level = [], False, 0
    for ln in lines:
        m = RE_HEADING.match(ln.strip())
        if m:
            h_level = len(m.group(1))
            if RE_TOC_TITLE.search(m.group(2)):
                skipping, level = True, h_level
                continue                      # 标题本身也一起删
            if skipping and h_level <= level:
                skipping = False
        if not skipping:
            out.append(ln)
    return out


def _drop_repeated_lines(lines, min_repeat=None, max_len=30):
    """
    反复出现且很短的整行 —— 典型是页眉/页脚/水印（"XX公司内部资料" 每页一次）。
    只删长度<=30 的行，跳过 markdown 结构行，避免误杀。
    阈值自适应：短文档（行数<120）出现 3 次即处理，长文档要 8 次。
    """
    if min_repeat is None:
        min_repeat = 3 if len(lines) < 120 else 8
    short = [ln.strip() for ln in lines if ln.strip() and len(ln.strip()) <= max_len]
    counter = Counter(short)
    noisy = {
        t for t, c in counter.items()
        if c >= min_repeat and not t.startswith(("#", "|", ">", "-", "*", "$", "```"))
        and "|" not in t
    }
    if not noisy:
        return lines
    return [ln for ln in lines if ln.strip() not in noisy]


def clean_markdown(raw: str, verbose=False):
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 1) HTML 残留：<br> 变换行，其余行内标签直接去掉（避免把"结算"拆成" 结算 "）
    text = RE_BR.sub("\n", text)
    text = RE_HTML_TAG.sub("", text)

    # 2) 图片：RAG 用不到二进制，转成 alt 文字（一般为空，等于删掉）
    text = RE_IMG.sub(lambda m: m.group(1).strip(), text)

    # 3) 控制字符
    text = RE_CTRL.sub("", text)

    lines = text.split("\n")

    # 4) 整个"目录"章节 + 页码行 / 目录行
    before = len(lines)
    lines = _drop_toc_section(lines)
    lines = [ln for ln in lines if not RE_PAGE_NO.match(ln) and not RE_TOC.match(ln)]
    dropped_noise = before - len(lines)

    # 5) 重复页眉页脚
    after_toc = len(lines)
    lines = _drop_repeated_lines(lines)
    dropped_repeat = after_toc - len(lines)

    text = "\n".join(lines)

    # 6) 收空白
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))
    text = RE_MULTI_SPACE.sub(" ", text)
    text = RE_BLANK.sub("\n\n", text).strip()

    if verbose:
        print(f"      清洗：删页码/目录行 {dropped_noise} 行，删重复页眉页脚 {dropped_repeat} 行，"
              f"字符数 {len(raw)} → {len(text)}")
    return text


# =====================================================================
# 四、切片
# =====================================================================
def split_markdown(md_text, source_name, chunk_size, overlap, page_lookup=None, header_prefix=False):
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    # 第一刀按标题切：保住章节内聚，chunk 不跨节
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
        strip_headers=False,
    )
    sections = header_splitter.split_text(md_text)

    # 第二刀按长度切：中文标点优先级，先断段落再断句再断字
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
    )
    chunks = char_splitter.split_documents(sections)

    docs = []
    for i, ch in enumerate(chunks):
        body = ch.page_content.strip()
        if len(re.sub(r"\s", "", body)) < MIN_CHUNK_CHARS:
            continue
        meta = dict(ch.metadata)
        path_parts = [meta[k] for k in ("h1", "h2", "h3", "h4") if meta.get(k)]
        prefix = " > ".join(path_parts)
        if header_prefix and prefix:
            body = f"【{prefix}】\n{body}"

        chunk_id = hashlib.md5(f"{source_name}|{i}|{body[:80]}".encode("utf-8")).hexdigest()[:24]
        docs.append({
            "chunk_id": chunk_id,
            "text": body,
            "source": source_name,
            "h1": meta.get("h1", ""),
            "h2": meta.get("h2", ""),
            "h3": meta.get("h3", ""),
            "header_path": prefix,
            "page": page_lookup(body) if page_lookup else -1,
            "chunk_index": i,
            "char_len": len(body),
        })
    return docs


def build_page_lookup(out_dir: Path):
    """
    从 MinerU 4.x 的 middle_json.json 反查页码（page_idx 0 基，+1 变页码）。
    结构是 pdf_info[i].blocks[j].lines[k].spans[m].content 这种嵌套，
    所以无脑递归 + 页码向下继承，不依赖具体 schema。
    找不到就返回 None，page 统一 -1，不影响检索。
    """
    files = sorted(out_dir.rglob("middle_json.json"))
    if not files:
        print("      [!] 没有 middle_json.json，page 记 -1（不影响检索，只是引用没页码）")
        return None

    try:
        data = json.loads(files[0].read_text("utf-8", errors="ignore"))
    except Exception as e:
        print(f"      [!] middle_json.json 解析失败（{e}），page 记 -1")
        return None

    pairs = []

    def walk(node, page):
        if isinstance(node, dict):
            if "page_idx" in node:
                try:
                    page = int(node["page_idx"])
                except (TypeError, ValueError):
                    pass
            for k in ("text", "content"):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    pairs.append((page, v))
            for v in node.values():
                if isinstance(v, (dict, list)):
                    walk(v, page)
        elif isinstance(node, list):
            for v in node:
                walk(v, page)

    walk(data, None)
    by_page = {}
    for p, t in pairs:
        if isinstance(p, int):
            by_page.setdefault(p, []).append(t)
    if not by_page:
        print("      [!] middle_json 里没找到页码字段，page 记 -1")
        return None

    flats = [(p, _norm("".join(v))) for p, v in by_page.items()]
    print(f"      页码映射：{files[0].name}，共 {len(flats)} 页")

    def lookup(chunk_text, _flats=flats):
        """阶梯匹配：先用长前缀精确命中，不行再短前缀兜（chunk 跨页/跨节时用得上）"""
        n = _norm(chunk_text)
        for length in (40, 24, 12):
            key = n[:length]
            if len(key) < 8:
                continue
            for p, page_text in _flats:
                if key in page_text:
                    return p + 1
        return -1

    return lookup


# =====================================================================
# 五、入库
# =====================================================================
def get_store(collection, drop_old=False):
    """
    langchain-milvus 0.3.x + pymilvus 2.6 的兼容坑：
    它用 MilvusClient._using 当 ORM 别名去 Collection(using=...)，而 pymilvus 2.6 的
    MilvusClient 只生成随机串（cm-xxxx）不注册 ORM 连接，于是已存在的集合一构造就
    ConnectionNotExistException。这里先自己注册一个固定别名的 ORM 连接，再把
    MilvusClient._using 指向它。
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
    return Milvus(
        embedding_function=embedding,
        connection_args={"uri": MILVUS_URI},
        builtin_function=BM25BuiltInFunction(analyzer_params={"type": "chinese"}),
        vector_field=["dense", "sparse"],
        collection_name=collection,
        auto_id=False,
        drop_old=drop_old,
    )


def purge_source(store, source_name):
    """删掉同一份文档的旧 chunk，保证重跑幂等"""
    expr = f'metadata["source"] == {json.dumps(source_name, ensure_ascii=False)}'
    try:
        col = getattr(store, "col", None)
        if col is not None:
            col.delete(expr=expr)
            col.flush()
        else:
            from pymilvus import MilvusClient
            MilvusClient(uri=MILVUS_URI).delete(collection_name=store.collection_name, filter=expr)
        print(f"      已清理旧数据：{source_name}")
    except Exception as e:
        print(f"      [!] 清理旧数据失败（首次建表时正常，忽略）：{e}")


def ingest(store, docs):
    from langchain_core.documents import Document

    lc_docs, ids = [], []
    for d in docs:
        lc_docs.append(Document(
            page_content=d["text"],
            metadata={
                "source": d["source"],
                "h1": d["h1"], "h2": d["h2"], "h3": d["h3"],
                "header_path": d["header_path"],
                "page": d["page"],
                "chunk_index": d["chunk_index"],
                "char_len": d["char_len"],
            },
        ))
        ids.append(d["chunk_id"])

    total = 0
    for i in tqdm(range(0, len(lc_docs), BATCH_SIZE), desc="入库"):
        batch = lc_docs[i:i + BATCH_SIZE]
        store.add_documents(batch, ids=ids[i:i + BATCH_SIZE])
        total += len(batch)
    return total


# =====================================================================
# 六、主流程
# =====================================================================
def main():
    ap = argparse.ArgumentParser("PDF -> MinerU 4.x -> md -> Milvus")
    ap.add_argument(
        "pdf",
        nargs="?",
        default=r"D:\PythonProjectAll\LLM\DeepLearning.pdf",
        help="PDF 路径（不传时使用默认路径）",
    )
    ap.add_argument("--md", help="跳过 MinerU，直接用现成的 md")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--tier", default="standard", choices=["flash", "basic", "standard", "advanced"])
    ap.add_argument("--pages", default="", help="只解析部分页，如 1-50 或 1-5,8（默认全本）")
    ap.add_argument("--mineru-cmd", default="", help="mineru-kit 路径（装在别的 conda 环境时用）")
    ap.add_argument("--header-prefix", action="store_true", help="把【h1>h2】拼进正文，提升召回")
    ap.add_argument("--dry-run", action="store_true", help="只解析切片，不入库")
    ap.add_argument("--recreate", action="store_true", help="删库重建（破坏性）")
    ap.add_argument("--yes", action="store_true", help="配合 --recreate，跳过二次确认")
    ap.add_argument("--out-root", default=str(MINERU_OUT_ROOT))
    args = ap.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF 不存在：{pdf_path}")

    # ---------- 1. 解析 ----------
    if args.md:
        md_path = Path(args.md).expanduser().resolve()
        if not md_path.is_file():
            raise SystemExit(f"md 不存在：{md_path}")
        mineru_out_dir = md_path.parent
        print(f"[1/5] 跳过 MinerU，直接读：{md_path}")
    else:
        md_path, mineru_out_dir = run_mineru(
            pdf_path, Path(args.out_root).expanduser(), args.tier, args.mineru_cmd, args.pages
        )

    # ---------- 2. 清洗 ----------
    print("[3/5] 清洗 Markdown…")
    clean_md = clean_markdown(md_path.read_text("utf-8", errors="ignore"), verbose=True)

    # ---------- 3. 切片 ----------
    print(f"[4/5] 切片（chunk={args.chunk_size}, overlap={args.overlap}）…")
    page_lookup = build_page_lookup(mineru_out_dir)
    docs = split_markdown(
        clean_md, pdf_path.name, args.chunk_size, args.overlap,
        page_lookup=page_lookup, header_prefix=args.header_prefix,
    )
    if not docs:
        raise SystemExit("切完 0 条，检查清洗是不是把内容删没了（加 --dry-run 看）")

    lens = sorted(d["char_len"] for d in docs)
    print(f"      共 {len(docs)} 条 | 长度 min/中位/max = "
          f"{lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")

    if args.dry_run:
        print("\n===== 前 3 条样例 =====")
        for d in docs[:3]:
            print(f"\n--- [{d['chunk_index']}] page={d['page']} "
                  f"path={d['header_path'] or '(无标题)'} len={d['char_len']} ---")
            print(d["text"][:400])
        print("\n[dry-run] 没有写库。确认没问题就去掉 --dry-run 正式入库。")
        return

    # ---------- 4. 入库 ----------
    print(f"[5/5] 写入 Milvus：{MILVUS_URI} / collection={args.collection}")
    if args.recreate:
        if not args.yes:
            ans = input(f"  [!] 将删除整个 collection「{args.collection}」及其它文档，确认？(yes/no) ")
            if ans.strip().lower() not in ("yes", "y"):
                raise SystemExit("已取消")
        print("      重建 collection…")

    store = get_store(args.collection, drop_old=args.recreate)
    purge_source(store, pdf_path.name)
    n = ingest(store, docs)
    print(f"\n[OK] 入库 {n} 条 -> collection「{args.collection}」")
    print(f"     下一步验证：python verify_rag.py --collection {args.collection} --query \"你的问题\"")


if __name__ == "__main__":
    main()
