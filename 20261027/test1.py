#大模型的访问 肯定是得走 网络 访问
#需要客户端

# openai

#访问线上的大模型 需要 ip地址 还需要apikey


from openai import OpenAI
import os

from langchain_openai import ChatOpenAI

client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key="sk-ws-H.PXXRYMP.qjID.MEYCIQCKfReT1gEDfo4unSaGqbmvN2-TLLP-0pIsu6A_hcTUqwIhAIESDDiS-h5s1BV599tHVAPcuOHlyvALZHWgAS88_-h4",
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
)

messages = [{"role": "user", "content": "你是谁"}]
completion = client.chat.completions.create(
    model="qwen3.8-flash",  # 您可以按需更换为其它深度思考模型
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True
)

# print(completion.choices[0].message.content)


is_answering = False  # 是否进入回复阶段
print("\n" + "=" * 20 + "思考过程" + "=" * 20)
for chunk in completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)