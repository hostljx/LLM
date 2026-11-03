#我们的单agnet 常见的构建方式  llm + tool + history

from langchain_openai import ChatOpenAI  #导入openai的模型
    
from tool_list import get_tool_list

#构建agent
from langchain.agents import create_agent

#agent 的记忆单元
from langgraph.checkpoint.memory import MemorySaver

 #1.构建模型
def load_model(top,temperature):
    return ChatOpenAI(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",api_key="1233",top_p=top,temperature=temperature)





 
# #2.构建工具

# tool_list = get_tool_list()

# #3.构建agent

# agent = create_agent(model=model,tools=tool_list,checkpointer=MemorySaver())


def get_agent(model,tool_list,system_prompt):
    return create_agent(model=model,tools=tool_list,system_prompt=system_prompt,checkpointer=MemorySaver())




def main():
    model = load_model(0.9,1)
    tool_list = get_tool_list()
    system_prompt = "你是一个天气助手，请根据用户的问题查询天气"

    agent = get_agent(model,tool_list,system_prompt)
    while True:
        user_input = input("请输入: ")
        if user_input == "exit":
            break
        response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print(response)



