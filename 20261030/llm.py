# -*- coding: utf-8 -*-
"""模型层：主模型负责对话，小模型负责压缩摘要，两条完全独立的链路。

- 主模型：远程 OpenAI 兼容接口（qwen3.8-max），按 token 消耗积分，默认不调用
- 小模型：本地 ollama qwen2.5:1.5b，压缩摘要用，不花一分钱

压缩是高频、低难度的任务，用本地 1.5B 完全够用，
没必要为了压缩摘要去消耗远程额度。
"""
import json
import urllib.request

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

import config


# ============================================================
# 主模型
# ============================================================
def build_main_model():
    """按给定参数构造主对话模型。

    extra_body 里关掉 thinking，避免推理过程占满 max_tokens。
    """
    return init_chat_model(
        model=config.MAIN_MODEL,
        base_url=config.BASE_URL,
        api_key=config.API_KEY,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        model_provider="openai",
        extra_body={"enable_thinking": False},
    )


class MockChatModel:
    """假主模型：不联网、不花积分，用来跑通链路。

    调 Redis、调压缩逻辑、验证上下文组装都用它，
    确认没问题了再加 --real 换真模型。
    """

    def invoke(self, messages):
        last = messages[-1].content if messages else ""
        # 把上下文规模报出来，方便观察压缩到底有没有生效
        return AIMessage(
            content=(
                f"[MOCK] 本次收到 {len(messages)} 条消息，"
                f"你刚说的是：{str(last)[:40]}。"
                f"（想看真实回复，启动时加 --real）"
            )
        )


# ============================================================
# 小模型（本地 ollama）
# ============================================================
def is_ollama_alive() -> bool:
    """探测 ollama 是否在跑。用于启动时给一句明确提示。"""
    try:
        urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def small_model_chat(system: str, user: str, json_mode: bool = True, timeout: int = 120) -> str:
    """调本地 ollama 的 /api/generate，返回纯文本响应。

    用 urllib 而不是 requests，少一个依赖。
    json_mode=True 时让 ollama 强制输出合法 JSON（qwen2.5 支持 format 参数）。
    """
    payload = {
        "model": config.SMALL_MODEL,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 500},
    }
    if json_mode:
        payload["format"] = "json"

    req = urllib.request.Request(
        f"{config.OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")
