#大模型结构化输出 工具所需要的参数的能力 叫做 functioncall/tool call

from langchain_ollama import ChatOllama
from langchain.tools import tool
from pydantic import BaseModel, Field

#1.构建模型
model = ChatOllama(model="qwen3.5:9b",base_url="http://192.168.1.105:11434")

#2.构建工具
class Weather(BaseModel):
    city: str = Field(description="城市名")

@tool(args_schema=Weather)
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""   #必须要有工具描述 工具描述详尽
    return f"{city}：晴，25°C"

#3.把工具给到大模型去看
model_with_tools = model.bind_tools([get_weather])

#4.大模型调用工具
res = model_with_tools.invoke("李白是谁")
print(res)

