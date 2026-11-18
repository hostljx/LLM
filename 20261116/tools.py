from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
import requests
import os
from dotenv import load_dotenv
from get_acctoken_id import get_acctoken_id, get_union_id
from pydantic import BaseModel, Field
from typing import Annotated
import datetime
load_dotenv()
DINGDING_ID = os.getenv("DINGDING_ID")
DINGDING_SECRET = os.getenv("DINGDING_SECRET")

BEIJING_TZ = datetime.timezone(datetime.timedelta(hours=8))


class CreateTodoTaskInput(BaseModel):
    subject: str = Field(description="待办任务标题，最大长度1024")
    description: str = Field(description="待办任务描述，最大长度1024")
    dueTime: int = Field(description="待办任务截止日期，Unix时间戳，单位毫秒",example="1617675000000")
    priority: int = Field(default=20,description="优先级，取值：10：较低20：普通30：较高40：紧急")

class DateTimeToTimestampInput(BaseModel):
    datetime_str: str = Field(
        description="北京时间，格式必须为 YYYY-MM-DD HH:MM:SS，例如 2026-07-05 17:00:00"
    )


@tool(args_schema=CreateTodoTaskInput)
def create_todo_task(subject: str, description: str, dueTime: int, priority: int,config: RunnableConfig) -> str:
    """
    创建待办任务
    """
    access_token = get_acctoken_id(DINGDING_ID, DINGDING_SECRET)
    union_id = get_union_id(access_token, config.get("configurable")["thread_id"])

    url = f"https://api.dingtalk.com/v1.0/todo/users/{union_id}/tasks"

    headers = {
        "Content-Type": "application/json",
        "x-acs-dingtalk-access-token": access_token
    }
    data = {
        "subject": subject,
        "description": description,
        "dueTime": dueTime,
        "priority": priority,
        "detailUrl" : {
            "appUrl" : "https://www.dingtalk.com",
            "pcUrl" : "https://www.dingtalk.com"
        }
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return "待办任务创建成功"
    else:
        return f"待办任务创建失败，错误信息：{response.json()}"



@tool(args_schema=DateTimeToTimestampInput)
def datetime_to_timestamp_ms(datetime_str: str) -> str:
    """将北京时间字符串转换为 Unix 毫秒时间戳，供 create_todo_task 的 dueTime 使用。"""
    try:
        dt = datetime.datetime.strptime(datetime_str.strip(), "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=BEIJING_TZ)
        timestamp_ms = int(dt.timestamp() * 1000)
        return str(timestamp_ms)
    except ValueError:
        return "转换失败：时间格式必须是 YYYY-MM-DD HH:MM:SS，例如 2026-07-05 17:00:00"
@tool
def get_current_beijing_time() -> str:
    """获取当前北京时间，用于判断用户说的日期是过去还是未来。"""
    now = datetime.datetime.now(BEIJING_TZ)
    return now.strftime("%Y-%m-%d %H:%M:%S")






def get_tools():
    return [create_todo_task, datetime_to_timestamp_ms, get_current_beijing_time]