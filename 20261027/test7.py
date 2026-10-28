import requests


url = "http://127.0.0.1:8000/users"

data = {
    "messages": [
        {"role": "system", "content": "你是一名专业的AI助手，你擅长提供准确、详细、全面的信息，你的回答必须严谨、专业、准确、详细、全面，不能有任何错误。"},
    ]
}

# response = requests.post(url, json=data)
# print(response.json())


def user_chat(messages):
    while True:
        input_message = input("请输入你的问题：")
        messages.append({"role": "user", "content": input_message})
        response = requests.post(url, json=data)
        print(response.json()["ai_content"])
        messages.append({"role": "assistant", "content": response.json()["ai_content"]})
        print("-"*100)


if __name__ == "__main__":
    user_chat(data["messages"])