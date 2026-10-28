# pip install langchain_openai
# pip install openai

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model

# 线上模型
# model  = ChatOpenAI(model="qwen3.8-flash",
#                     base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
#                     api_key="sk-ws-H.YHLYEMH.Ty0z.MEQCICEHcNQatDrpzRTxcH1GdQQ694goJglv7h-ElnSNgX-aAiAlQv9Xi9_Q5ANih60eCDYY4oH0KDx5SGN372fTDOPj0w",
#                     max_tokens=1000)
# messages = [{"role": "user", "content": "你是谁"}]


# response = model.invoke(messages)
# print(type(response))
# print(response.content)

#本地模型 调用ollma


# model = ChatOpenAI(model="qwen3.5:9b",
#                     base_url="http://192.168.1.105:11434/v1",
#                     api_key="123",
#                     max_tokens=1000)

# messages = [{"role": "user", "content": "你是谁"}]


# response = model.invoke(messages)
# print(type(response))
# print(response.content)



# model = ChatOllama(model="qwen3.5:9b",
#                     base_url="http://192.168.1.105:11434",
#                     max_tokens=1000)

# messages = [{"role": "user", "content": "你是谁"}]

# response = model.invoke(messages)
# print(type(response))
# print(response.content)



#使用init_chat_model
model = init_chat_model(
    model="qwen2.5:1.5b",
    base_url="http://localhost:11434/v1",
    api_key="123",
    max_tokens=1000,
    model_provider="openai"
)

messages = [{"role": "user", "content": "你是谁"}]
response = model.invoke(messages)
print(type(response))
print(response.content)


