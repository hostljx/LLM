# -*- coding: utf-8 -*-
"""集中配置。

所有可调参数都在这里，改配置不用动业务代码。
全部支持环境变量覆盖，方便换 key 或换模型。
"""
import os

# ========== 主模型（远程接口，会消耗积分）==========
MAIN_MODEL = os.getenv("MAIN_MODEL", "qwen3.8-max")
BASE_URL = os.getenv("BASE_URL", "https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1")
API_KEY = os.getenv("API_KEY", "sk-ws-H.YIRMMIE.hrIq.MEUCIB-_MLZKO9KoGhb3R1S6RSk-tmqNG2khfl-h-GgEerArAiEApceh4mAjzMKlflD8jUs47tl9HeZeAS1NNeQKGZ5G7t4")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# 默认关闭真实调用：跑通链路、调试 Redis 逻辑时不该烧积分。
# 想真的聊天，命令行加 --real。
USE_REAL_LLM = os.getenv("USE_REAL_LLM", "0") == "1"

# ========== 小模型（本地 ollama，0 积分）==========
# 压缩摘要这种任务用 1.5B 足够，没必要浪费远程额度。
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SMALL_MODEL = os.getenv("SMALL_MODEL", "qwen2.5:1.5b")

# ========== Redis ==========
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL = 7 * 24 * 3600          # 会话 7 天过期

# ========== 压缩策略 ==========
# 真实场景按主模型上下文窗口的 50%~60% 设；这里调小是为了方便观察压缩发生。
TOKEN_BUDGET = 1500
KEEP_TURNS = 2                        # 最近 2 轮（4 条消息）原文不压，保证当前对话连贯
MIN_BATCH = 4                         # 待压消息少于 4 条就不值得调模型
COMPACT_EVERY_N_TURNS = 12            # 消息条数兜底：每 12 条强制压一次
                                      # 注意 turns 统计的是消息条数（一问一答算 2），不是对话轮次
LOCK_TTL_MS = 60_000                  # 压缩锁超时，防止 worker 挂了锁不释放
