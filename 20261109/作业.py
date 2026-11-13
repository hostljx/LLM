# -*- coding: utf-8 -*-
"""
RAG 以「工具调用(Tool Calling)」的方式接入大模型

核心思路：
    1. 用 bge-m3 本地向量模型，把已有知识库(Chroma) 当作一个「检索工具」。
    2. 用 @tool 装饰器把检索逻辑封装成一个 LangChain Tool。
    3. 把这个 Tool 绑定到 Ollama 大模型，让模型在回答问题时，
       如果发现需要知识库，就自动调用该工具检索，再把结果融合进最终回答。

依赖：
    pip install langchain langchain-ollama langchain-community langchain-core
    pip install chromadb pymupdf
    ollama pull bge-m3          # 嵌入模型
    ollama pull qwen2.5:7b      # 大模型(可换成你本地已有的任何对话模型)
"""

# ============ 1. 加载嵌入模型 + 打开已保存的向量库 ============

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# bge-m3 嵌入模型：负责把「问题文本」转成向量，用来和库里的文档向量做相似度比对
embedding_model = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://127.0.0.1:11434",
    client_kwargs={"trust_env": False},
)

# 打开之前已经保存好的向量数据库（注意要和 saveVector.py 里的 persist_directory 一致）
VECTOR_DB_PATH = "D:/PythonProjectAll/LLM/20261109/chroma_db"
COLLECTION_NAME = "car_info"   # 集合名称，要和建库时用的一致

db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    persist_directory=VECTOR_DB_PATH,
)


# ============ 2. 把「检索」封装成 Tool ============

from langchain_core.tools import tool


@tool
def search_knowledge_base(query: str) -> str:
    """在本地知识库中检索与 query 相关的文档片段。

    当用户的问题需要依靠知识库里的资料才能回答时，调用本工具进行检索。
    参数 query 就是用户想要查询的问题或关键词。
    返回与问题最相关的若干文档片段文本。
    """
    # k=4 表示取最相似的 4 个片段，可以根据需要调整
    docs = db.similarity_search(query, k=4)

    if not docs:
        return "知识库中没有检索到相关内容。"

    # 把检索到的片段拼接成一段文本返回给大模型，作为回答的依据
    results = []
    for i, doc in enumerate(docs, start=1):
        results.append(f"【片段 {i}】\n{doc.page_content}")
    return "\n\n".join(results)


# 工具列表：以后想加更多工具(如查天气、算数)，往这个列表里追加即可
tools = [search_knowledge_base]


# ============ 3. 把工具绑定到 Ollama 大模型 ============

from langchain_ollama import ChatOllama

# 对话大模型：负责理解问题、决定是否调用工具、并根据工具返回的结果生成最终回答
llm = ChatOllama(
    model="qwen2.5:7b",              # 换成你本地 ollama 里已 pull 的对话模型
    base_url="http://127.0.0.1:11434",
    temperature=0,                    # 0 让回答更稳定，减少胡编
)

# bind_tools：让大模型"知道"自己有哪些工具可用，需要时它会在输出里给出工具调用请求
llm_with_tools = llm.bind_tools(tools)


# ============ 4. 组装「Agent 执行链」：模型 <-> 工具 的闭环 ============

from langchain_core.messages import HumanMessage
from langchain_core.messages.tool import ToolMessage


def run_rag_agent(question: str) -> str:
    """问一次问题：模型决定是否检索 -> 执行检索 -> 用检索结果生成最终回答。"""

    # 先把用户问题交给带工具的大模型，看它是否发起工具调用
    messages = [HumanMessage(content=question)]
    ai_msg = llm_with_tools.invoke(messages)

    # 如果模型发起了工具调用，就逐个执行工具，并把结果回填给它
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # 根据工具名找到对应的工具并执行
        tool_obj = {t.name: t for t in tools}[tool_name]
        tool_result = tool_obj.invoke(tool_args)

        # 把工具执行结果作为 ToolMessage 追加到对话里
        messages.append(ai_msg)
        messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
        )
    # 再次调用模型，让它综合检索到的资料，生成最终回答
    final_msg = llm_with_tools.invoke(messages)
    return final_msg.content


# ============ 5. 运行示例 ============

if __name__ == "__main__":
    # 换成你想问的问题即可，模型会自动判断是否需要检索知识库
    question = "帮我介绍一下这份资料里关于汽车的信息"
    answer = run_rag_agent(question)

    print("=" * 50)
    print("问题：", question)
    print("=" * 50)
    print("回答：")
    print(answer)
