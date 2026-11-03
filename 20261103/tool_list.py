from pydantic import BaseModel, Field
from langchain.tools import tool

class Weather(BaseModel):
    city: str = Field(description="城市名")

@tool(args_schema=Weather)
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    return f"{city}：晴，25°C"



def get_tool_list():
    return [get_weather]

