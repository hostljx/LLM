"""集中配置。

所有配置都来自环境变量或项目根目录的 .env。
钉钉 AppKey / AppSecret 留空也能本地跑工具测试，只有真正连钉钉时才要求填写。

几个时间参数之间的关系（不要随意改乱）：
    agent_timeout < processing_ttl < claim_min_idle_ms / 1000
保证正在处理的消息不会在锁有效期内被巡检线程误抢。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    # ===== 钉钉开放平台 =====
    dingtalk_app_key: str
    dingtalk_app_secret: str
    dingtalk_robot_code: str
    dingtalk_api_base: str
    dingtalk_trust_env: bool
    dingtalk_timeout: float
    dingtalk_reply_retries: int
    dingtalk_reply_max_chars: int

    # ===== LLM（OpenAI 兼容） =====
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout: float
    llm_max_tool_rounds: int
    llm_temperature: float
    # 本机 Ollama 必须绕过系统代理（Windows 上 httpx 会读系统代理，导致 502）；
    # 换成需要代理的云端 API 时可设为 true。
    llm_trust_env: bool

    # ===== 天气（和风天气 QWeather） =====
    weather_api_key: str
    weather_api_base: str
    weather_timeout: float
    weather_trust_env: bool

    # ===== Redis =====
    redis_url: str
    key_prefix: str
    queue_maxlen: int
    dedup_ttl: int
    done_ttl: int
    answer_ttl: int
    processing_ttl: int
    reply_ttl: int
    conversation_lock_ttl: int
    conversation_lock_wait: float
    claim_min_idle_ms: int
    retry_base_seconds: float
    max_attempts: int
    agent_timeout: float
    history_limit: int
    history_ttl: int

    # ===== Worker =====
    max_concurrency: int

    # ---- Redis key 统一在这里生成 ----
    @property
    def stream_key(self) -> str:
        return f"{self.key_prefix}:stream"

    @property
    def group_name(self) -> str:
        return f"{self.key_prefix}:workers"

    @property
    def dlq_key(self) -> str:
        return f"{self.key_prefix}:dlq"

    @property
    def retry_key(self) -> str:
        return f"{self.key_prefix}:retry"

    def require_dingtalk(self) -> None:
        """连接钉钉前调用；凭证没填就给出明确提示。"""
        missing = [
            name
            for name, value in (
                ("DINGTALK_APP_KEY", self.dingtalk_app_key),
                ("DINGTALK_APP_SECRET", self.dingtalk_app_secret),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "缺少钉钉应用凭证: "
                + ", ".join(missing)
                + "。请在项目根目录的 .env 中填写后重试。"
            )


settings = Settings(
    # 钉钉
    dingtalk_app_key=_str("DINGTALK_APP_KEY"),
    dingtalk_app_secret=_str("DINGTALK_APP_SECRET"),
    dingtalk_robot_code=_str("DINGTALK_ROBOT_CODE"),
    dingtalk_api_base=_str("DINGTALK_API_BASE", "https://api.dingtalk.com"),
    dingtalk_trust_env=_bool("DINGTALK_TRUST_ENV", True),
    dingtalk_timeout=_float("DINGTALK_TIMEOUT", 15.0),
    dingtalk_reply_retries=_int("DINGTALK_REPLY_RETRIES", 3),
    dingtalk_reply_max_chars=_int("DINGTALK_REPLY_MAX_CHARS", 3500),
    # LLM
    llm_base_url=_str("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
    llm_api_key=_str("LLM_API_KEY", "ollama"),
    llm_model=_str("LLM_MODEL", "qwen3.5:9b"),
    llm_timeout=_float("LLM_TIMEOUT", 120.0),
    llm_max_tool_rounds=_int("LLM_MAX_TOOL_ROUNDS", 6),
    llm_temperature=_float("LLM_TEMPERATURE", 0.3),
    llm_trust_env=_bool("LLM_TRUST_ENV", False),
    # 天气（和风天气）
    weather_api_key=_str("WEATHER_API_KEY"),
    weather_api_base=_str(
        "WEATHER_API_BASE", "https://devapi.qweather.com"
    ).rstrip("/"),
    weather_timeout=_float("WEATHER_TIMEOUT", 10.0),
    weather_trust_env=_bool("WEATHER_TRUST_ENV", True),
    # Redis / 队列
    redis_url=_str("REDIS_URL", "redis://localhost:6379/1"),
    key_prefix=_str("KEY_PREFIX", "robot"),
    queue_maxlen=_int("QUEUE_MAXLEN", 100000),
    dedup_ttl=_int("DEDUP_TTL", 86400),
    done_ttl=_int("DONE_TTL", 86400),
    answer_ttl=_int("ANSWER_TTL", 3600),
    processing_ttl=_int("PROCESSING_TTL", 300),
    reply_ttl=_int("REPLY_TTL", 120),
    conversation_lock_ttl=_int("CONVERSATION_LOCK_TTL", 300),
    conversation_lock_wait=_float("CONVERSATION_LOCK_WAIT", 60.0),
    claim_min_idle_ms=_int("CLAIM_MIN_IDLE_MS", 360000),
    retry_base_seconds=_float("RETRY_BASE_SECONDS", 3.0),
    max_attempts=_int("MAX_ATTEMPTS", 3),
    agent_timeout=_float("AGENT_TIMEOUT", 280.0),
    history_limit=_int("HISTORY_LIMIT", 20),
    history_ttl=_int("HISTORY_TTL", 604800),
    # Worker
    max_concurrency=_int("MAX_CONCURRENCY", 4),
)
