"""
多轮日常对话助手（LangChain + 本地 Ollama）。

普通聊天直接回答；当用户表达出查询天气的意图时，调用 weather_tool.py 里的
get_today_weather 工具拿到真实天气，再结合对话历史回答。

运行：
    python chat.py
"""

import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from weather_tool import get_today_weather

# 本地 Ollama 配置（用 ollama list 查看本机已安装的模型，按需修改）
OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

# 保留最近多少条历史消息（一问一答算 2 条）
MAX_HISTORY = 8


def build_llm():
    """默认用本地 Ollama；如需在线 API，设置环境变量 LLM_PROVIDER=openai 并填好下面的占位配置。"""
    provider = os.getenv("LLM_PROVIDER", "ollama")

    if provider == "ollama":
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    # ---------------- 在线 API 占位 ----------------
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="gpt-4o-mini",              # 占位：换成要用的模型
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),  # 占位：换成真实 Key
        base_url="https://api.openai.com/v1",           # 占位：换成实际接口地址
    )


# 判断用户是不是在问天气的关键词（小模型做意图分类不稳定，先用关键词把普通聊天挡掉）
WEATHER_KEYWORDS = (
    "天气", "气温", "温度", "下雨", "降雨", "下雪", "降雪", "预报",
    "紫外线", "湿度", "风速", "几度", "多少度", "穿什么", "带伞",
)

# 从用户的话里提取城市名
EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "提取用户这句话里要查询天气的城市名，只输出城市名；没有城市就输出：无。"),
    ("human", "{question}"),
])

# 普通聊天
CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的中文助手，和用户自然聊天，回答简洁。"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

# 带真实天气数据的回答
WEATHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个友好的中文助手。下面提供了刚查询到的真实天气数据，"
     "请结合对话历史回答用户，只能依据这份数据，不要编造；"
     "如果数据是错误信息，就如实告诉用户。"),
    MessagesPlaceholder("history"),
    ("human", "{question}\n\n【刚查到的天气数据】\n{weather}"),
])


def _clean_city(text: str) -> str:
    """清理模型输出，返回城市名或空字符串。"""
    city = text.strip().strip("。！？：:；;，, \"'“”")
    if city in ("", "无", "没有", "无城市"):
        return ""
    # 太长或带空格/换行，基本不是城市名，按“没识别到”处理
    if len(city) > 10 or any(ch.isspace() for ch in city):
        return ""
    return city


def main():
    llm = build_llm()
    extract_city = EXTRACT_PROMPT | llm | StrOutputParser()
    chat = CHAT_PROMPT | llm | StrOutputParser()
    answer_weather = WEATHER_PROMPT | llm | StrOutputParser()

    history = []
    last_was_weather = False  # 上一轮是不是在聊天气（用于“那北京呢”这种追问）
    print("助手已启动，可以正常聊天；想查天气直接说城市名（输入 exit 退出）。")

    while True:
        question = input("\n你：").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        # 1. 判断意图：当前句带天气关键词；或上一轮在聊天气且这句很短（追问城市）
        wants_weather = any(word in question for word in WEATHER_KEYWORDS)
        city = ""
        if wants_weather or (last_was_weather and len(question) <= 6):
            city = _clean_city(extract_city.invoke({"question": question}))
            # 上一轮是天气，但这句没认出城市（比如“谢谢”），仍然按普通聊天处理
            wants_weather = wants_weather or bool(city)

        # 2. 想查天气但没说城市：先问清楚
        if wants_weather and not city:
            answer = "你想查哪个城市的天气？例如“成都天气”。"
            last_was_weather = False
        # 3. 天气查询：调工具拿真实数据，再结合历史回答
        elif wants_weather:
            data = get_today_weather.invoke({"city": city})
            if "天气：" not in data:
                print("（接口返回：" + data + "）")
            answer = answer_weather.invoke({"history": history, "question": question, "weather": data})
            last_was_weather = True
        # 4. 普通聊天
        else:
            answer = chat.invoke({"history": history, "question": question})
            last_was_weather = False

        print("助手：" + answer)

        # 5. 记录多轮对话历史
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        history = history[-MAX_HISTORY:]


if __name__ == "__main__":
    main()
