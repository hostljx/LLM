from langchain.chat_models import init_chat_model


model = init_chat_model(
    model="qwen3.8-flash",
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    api_key="sk-ws-H.PXXRYMP.qjID.MEYCIQCKfReT1gEDfo4unSaGqbmvN2-TLLP-0pIsu6A_hcTUqwIhAIESDDiS-h5s1BV599tHVAPcuOHlyvALZHWgAS88_-h4",
    max_tokens=1000,
    temperature=1.2,
    top_p=0.9,
    model_provider="openai"
)


messages = [
    {"role": "system", "content": "你是一名文学家，你擅长诗词歌赋，你的回答必须温文尔雅，用词生动"},
]


#多轮对话


def multi_turn_conversation(messages):
    while True:
        input_message = input("请输入你的问题：")
        messages.append({"role": "user", "content": input_message})
        response = model.invoke(messages)
        print(response.content)
        messages.append(response)
        print("-"*100)


if __name__ == "__main__":
    multi_turn_conversation(messages)