#大模型工具构建的方式

from langchain.tools import tool
from pydantic import BaseModel, Field


#简单的工具构建方案
# `@tool
# def get_weather(city: str) -> str:
#     """查询指定城市的当前天气。"""   #必须要有工具描述 工具描述详尽
#     return f"{city}：晴，25°C"

#复杂工具构建方案

class Weather(BaseModel):
    city: str = Field(description="城市名")

@tool(args_schema=Weather)
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""   #必须要有工具描述 工具描述详尽
    return f"{city}：晴，25°C"