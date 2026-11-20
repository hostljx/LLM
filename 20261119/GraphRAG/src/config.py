"""
全局配置与客户端封装
功能：
1. 从项目根目录的 .env 文件加载密钥等敏感配置（避免硬编码在代码里）
2. 封装 DeepSeek 大模型调用（OpenAI 兼容协议）
3. 封装 Neo4j 数据库驱动
其余所有脚本都从本模块导入 chat() 和 get_driver()，保证配置只写一处。

注意： 以下的单例模式都是非线程安全的。 但是我们课程演示中没有问题。
double check
"""
import os
import json
import re
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# -------------- 配置相关路径 ---------------
# 项目根目录 = src 目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "sanguo.txt"  # 原始语料
OUTPUT_DIR = PROJECT_ROOT / "output"  # 中间结果输出目录
OUTPUT_DIR.mkdir(exist_ok=True)  # 不存在则自动创建

# -------------- 加载.env ---------------
load_dotenv(PROJECT_ROOT / ".env")

# ---------- DeepSeek 配置 ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# ---------- 关闭思考模式 ----------
# 不同厂商的"关思考"参数名不一样，写错会被服务端拒绝或静默忽略：
#   DeepSeek: extra_body={"thinking": {"type": "disabled"}}
#   阿里百炼 Qwen: extra_body={"enable_thinking": False}
# qwen3.8-max 默认可能开启深度推理，抽取/摘要这类任务不需要，关掉能省大量 token。
_IS_QWEN = "dashscope" in DEEPSEEK_BASE_URL or "aliyuncs" in DEEPSEEK_BASE_URL
_THINK_OFF = {"enable_thinking": False} if _IS_QWEN else {"thinking": {"type": "disabled"}}

# ---------- Neo4j 配置 ----------
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ---------- 本地 Embedding 模型配置 ----------
# bge-small-zh-v1.5：中文向量模型，512 维，CPU 即可秒级推理
# 用于实体链接的"向量语义层"
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
EMBEDDING_DIM = 512
# bge 官方建议：检索类"查询"需加指令前缀（文档/段落侧不加），能显著提升检索精度
# https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 全局单例：LLM 客户端 & Neo4j 驱动 & Embedding 模型（进程内复用）
_llm_client = None
_neo4j_driver = None
_embedding_model = None

def get_llm_client():
    """返回 DeepSeek 客户端（懒加载单例）。"""
    global _llm_client
    if _llm_client is None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，请先在.env中填写")
        _llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    return _llm_client


def chat(prompt: str, system: str = "", json_mode: bool = False,
         temperature: float = 0.3) -> str:
    """
    调用 DeepSeek 完成一次对话，返回模型输出的文本。

    参数说明：
    - prompt:      用户输入（本课程所有任务都是单轮对话）
    - system:      系统提示词，用于约束模型行为
    - json_mode:   True 时启用 DeepSeek 的 JSON Output 模式，
                   强制模型输出合法 JSON（实体抽取时非常关键）
    - temperature: 采样温度。抽取类任务用低温度(0.3)保证稳定，
                   摘要/问答可适当调高
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        # 关闭思考模式（DeepSeek / Qwen 参数名不同，见文件上方 _THINK_OFF）
        "extra_body": _THINK_OFF,
    }
    if json_mode:
        # DeepSeek 官方 JSON Output 文档要求：
        # 启用 response_format={'type': 'json_object'} 时，
        # prompt 中必须包含 "json" 字样并给出期望格式示例，否则可能返回空内容。
        kwargs["response_format"] = {"type": "json_object"}

    response = get_llm_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content

def parse_json_safely(text: str):
    """
    容错的 JSON 解析：
    即便开了 JSON 模式，健壮的工程代码也应该兜底处理异常输出。
    依次尝试：1) 直接解析  2) 去掉 markdown 代码围栏后解析
             3) 正则截取第一个 { 到最后一个 } 之间的内容解析
    全部失败则返回 None，由调用方决定跳过或重试。
    """
    # 尝试 1：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试 2：去掉 ```json ... ``` 代码围栏
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 尝试 3：截取最外层花括号
    # python中正则符号. 默认是不能匹配\n
    # re.DOTALL表示.可以匹配\n了
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

def get_driver():
    """返回 Neo4j 驱动（懒加载单例）。用完整个程序后无需手动关闭，进程退出自动释放。"""
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            uri=NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            # 关闭服务端通知回显（如"过程即将弃用"的 WARNING），保持课堂输出干净
            notifications_min_severity="OFF"
        )
        _neo4j_driver.verify_connectivity() # 立即验证连接，配置错误时尽早报错
    return _neo4j_driver

def get_embedding_model():
    """返回本地 bge 向量模型（懒加载单例，首次调用加载约 2 秒）。"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer  # 延迟导入：不用向量功能就不加载
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    return _embedding_model

def embed_texts(texts: list[str], is_query: bool = False):
    """
    把一批文本编码为归一化向量（512 维）。

    - is_query=True：编码"用户问题"时使用，自动加 bge 官方检索指令前缀
    - is_query=False：编码"实体描述"等文档侧内容，不加前缀
    归一化后向量点积 = 余弦相似度，与 Neo4j 向量索引的 cosine 配置对应。
    """
    if is_query:
        texts = [BGE_QUERY_INSTRUCTION + t for t in texts]
    vectors = get_embedding_model().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]   # numpy 转成 Python list，便于传给 Neo4j


def run_cypher(query: str, **params):
    """执行一条 Cypher 语句并返回所有记录（list[dict]）。教学用的极简封装。"""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, **params)
        return [record.data() for record in result]
