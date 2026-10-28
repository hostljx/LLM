import asyncio
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from langchain.chat_models import init_chat_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("收到请求")
logger.warning("模型返回空内容")
logger.error("调用失败")


model = init_chat_model(
    model="qwen3.8-flash",
    base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",
    api_key="sk-ws-H.YHLYEMH.Ty0z.MEQCICEHcNQatDrpzRTxcH1GdQQ694goJglv7h-ElnSNgX-aAiAlQv9Xi9_Q5ANih60eCDYY4oH0KDx5SGN372fTDOPj0w",
    max_tokens=1000,
    temperature=1.2,
    top_p=0.9,
    model_provider="openai"
)

model2 = init_chat_model(
    model="qwen2.5:1.5b",
    base_url="http://localhost:11434/v1",
    api_key="123",
    max_tokens=1000,
    model_provider="openai"
)


app = FastAPI()

'''
# 容错处理流程：
# 1. 先解析请求体，并检查 messages 是否为非空列表。
# 2. 优先调用线上主模型，遇到网络错误、超时、异常或空响应时自动重试。
# 3. 每次重试之间逐步增加等待时间，避免短时间内重复请求造成压力。
# 4. 主模型多次失败后，切换到本地备用模型并再次进行有限次数的重试。
# 5. 两个模型都不可用时返回 503，明确告知客户端暂时无法提供服务。
#
# 常见 HTTP 状态码：
# 200：请求成功；400：请求体格式错误；404：请求路径不存在；
# 405：请求方法不允许；408：请求超时；413：请求体过大；
# 422：请求参数校验失败；429：请求过于频繁；
# 500：服务器内部错误；502：上游模型服务异常；503：模型服务不可用。
#
# invoke_with_retry 会统一处理模型调用异常和响应内容校验：
# - 字符串响应会去除首尾空白，空字符串会被视为空响应。
# - 列表等非字符串内容只要非空，也视为有效响应。
# - 网络类异常使用 warning 记录，其他未预期异常使用 exception 记录堆栈。
'''
@app.post("/users")
async def user_chat(request: Request):
    try:
        data = await request.json()
    except ValueError as exc:
        logger.warning("请求体不是合法 JSON")
        raise HTTPException(status_code=400, detail="请求体必须是合法 JSON") from exc

    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=422, detail="messages 必须是非空列表")

    response = await invoke_with_retry(model, messages, "模型1", attempts=3)
    if response is None:
        logger.warning("模型1多次调用失败，切换到备用模型")
        response = await invoke_with_retry(model2, messages, "备用模型", attempts=2)

    if response is None:
        logger.error("主模型和备用模型均调用失败")
        raise HTTPException(
            status_code=503,
            detail="当前没有可用的模型服务，请稍后重试",
        )

    return {"ai_content": response}


def _content_from_response(response: Any) -> Any:
    """Return non-empty model content while supporting string and block content."""
    content = getattr(response, "content", None)
    if content is None:
        return None
    if isinstance(content, str):
        return content.strip() or None
    return content if content else None


async def invoke_with_retry(
    chat_model: Any, messages: list, name: str, attempts: int
) -> Any:
    for attempt in range(1, attempts + 1):
        try:
            response = chat_model.invoke(messages)
            content = _content_from_response(response)
            if content is not None:
                logger.info("%s调用成功，尝试次数为%d", name, attempt)
                return content
            logger.warning("%s返回空内容，第%d次尝试", name, attempt)
        except (TimeoutError, ConnectionError, OSError) as exc:
            logger.warning(
                "%s网络或超时错误，第%d/%d次尝试：%s",
                name,
                attempt,
                attempts,
                exc,
            )
        except Exception:
            logger.exception(
                "%s调用出现未预期错误，第%d/%d次尝试",
                name,
                attempt,
                attempts,
            )

        if attempt < attempts:
            await asyncio.sleep(2 ** (attempt - 1))

    return None


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)