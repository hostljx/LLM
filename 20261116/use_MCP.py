#pip install langchain_mcp_adapters


#MCP 只能异步使用


from langchain_mcp_adapters.client import MultiServerMCPClient

import asyncio

from langchain_ollama import  ChatOllama

from langchain.agents import create_agent


model1 = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434",reasoning= False)


mcp_server = MultiServerMCPClient(
    {
    # "12306-mcp": {
    #   "transport": "sse",
    #   "url": "https://mcp.api-inference.modelscope.net/0c79cd33e22e44/sse"
    # },
    "test": {
      "transport": "sse",
      "url": "http://192.168.2.197:19530/sse"
    }
    }
)



async def create_agent_mcp():

    tools = await mcp_server.get_tools()
    print(tools)

    # agent = create_agent(model1, tools)
    # result = await agent.ainvoke({"messages":[("user","成都到北京的火车最早的是几点?")]})
    # print(result)



if __name__ == "__main__":
    asyncio.run(create_agent_mcp())