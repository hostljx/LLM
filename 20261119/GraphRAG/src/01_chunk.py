"""
文本分块（chunk）

"""
import json

from config import DATA_FILE, OUTPUT_DIR

CHUNK_SIZE = 600   # 每块最大字符数
OVERLAP = 100      # 相邻块之间的重叠字符数

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    """把长文本切分成带重叠的 chunk 列表。"""
    # 1) 按空行切分出自然段，去掉首尾空白和空段
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""  # 正在累积的当前块内容

    for para in paragraphs:
        # 2) 若加入这个段落后超出上限，则先封存当前块
        if current and len(current) + len(para) + 1 > chunk_size:
            chunks.append(current)
            # 3) 从上一块的末尾截取 overlap 个字符，作为新块的开头（重叠区）
            #    这样跨块的句子/上下文不会被硬生生切断
            current = current[-overlap:] + "\n" + para
        else:
            # 未超限则继续累积（块内段落之间用换行分隔）
            current = current + "\n" + para if current else para

    if current:                 # 别忘了最后一块
        chunks.append(current)

    # 4) 包装成带 id 的字典，id 用于后续在 Neo4j 中做来源追溯
    return [{"id": f"chunk-{i}", "text": c} for i, c in enumerate(chunks)]

def main():
    # 读取原始语料
    text = DATA_FILE.read_text(encoding="utf-8")
    chunks = split_text(text)

    # 保存中间结果，供下一步 extract 使用
    out_file = OUTPUT_DIR / "chunks.json"
    out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印统计信息
    print(f"原文总长度: {len(text)} 字符")
    print(f"切分为 {len(chunks)} 个 chunk（上限 {CHUNK_SIZE} 字，重叠 {OVERLAP} 字）")
    for c in chunks[:3]:
        print(f"  {c['id']}: {len(c['text'])} 字 | 开头: {c['text'][:30]}...")
    print(f"已保存到 {out_file}")
    return chunks


if __name__ == "__main__":
    main()