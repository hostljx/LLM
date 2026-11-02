# -*- coding: utf-8 -*-
"""会话存储层：只负责和 Redis 打交道，不写任何业务逻辑。

Redis 里一个会话拆成四份，各管一件事：

    sess:{sid}:msgs      Stream   原始对话，一条一条追加
    sess:{sid}:summary   String   滚动摘要，每次压缩后覆盖
    sess:{sid}:facts     Hash     硬事实（单号、金额、日期、状态），只增不压
    sess:{sid}:meta      Hash     tokens / turns / last_compacted_id / summary_ver
    sess:{sid}:lock      String   压缩互斥锁

为什么 msgs 用 Stream 而不是 List：
Stream 的消息 ID 是单调递增的，last_compacted_id 天然就是"已压到哪条"的游标。
List 用 LTRIM 删完就不知道压到哪了，压缩失败重启会重复压或漏压。
"""
import time

import redis

import config


def _key(sid: str, suffix: str) -> str:
    """统一拼 key，避免各处手写字符串拼错。"""
    return f"sess:{sid}:{suffix}"


def connect():
    """连接 Redis。连不上直接报错退出 —— 本项目要求用真实 Redis，不做内存降级。

    这样"会话存在 Redis 里"才是真的成立（可持久化、可被多个进程共享、
    重启不丢），而不是进程内的一份内存副本。
    """
    try:
        r = redis.from_url(config.REDIS_URL, decode_responses=True, socket_connect_timeout=3)
        r.ping()
        ver = r.info("server").get("redis_version", "?")
        print(f"[store] 已连接 Redis {config.REDIS_URL}  (v{ver})")
        return r
    except Exception as exc:
        raise SystemExit(
            f"\n[store] 连不上 Redis：{exc}\n"
            f"        地址 {config.REDIS_URL}\n\n"
            "先启动 Redis，二选一：\n"
            "  1) 双击 D:\\PythonProjectAll\\LLM\\20261030\\start_redis.bat\n"
            "  2) 命令行：\n"
            "     cd D:\\PythonProjectAll\\LLM\\20261030\\redis-bin\\Redis-8.10.3-Windows-x64-msys2\n"
            "     redis-server.exe redis.conf\n"
        )


def estimate_tokens(text: str) -> int:
    """粗估 token 数：中文按 1 字 1 token，非中文按 4 字符 1 token。

    只是用来判断该不该压缩，不需要精确，所以不引入 tiktoken 这种重依赖。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk + (len(text) - cjk) // 4 + 1


def append(r, sid: str, role: str, content: str, tokens: int) -> str:
    """追加一条消息，并累计 token / 轮数。返回消息 ID（作为后续游标）。"""
    mid = r.xadd(_key(sid, "msgs"), {
        "role": role,
        "c": content,
        "ts": f"{time.time():.3f}",
    })

    meta = _key(sid, "meta")
    r.hincrby(meta, "tokens", int(tokens))
    r.hincrby(meta, "turns", 1)
    r.hset(meta, "last_active", f"{time.time():.3f}")

    # 四个 key 一起续期，避免只续了一个导致数据半死不活
    for suffix in ("msgs", "meta", "summary", "facts"):
        r.expire(_key(sid, suffix), config.SESSION_TTL)
    return mid


def get_meta(r, sid: str) -> dict:
    """读会话元信息，顺手把字符串转成 int/float。"""
    m = r.hgetall(_key(sid, "meta"))
    return {
        "tokens": int(m.get("tokens", 0) or 0),
        "turns": int(m.get("turns", 0) or 0),
        "last_compacted_id": m.get("last_compacted_id", "0-0") or "0-0",
        "summary_ver": int(m.get("summary_ver", 0) or 0),
        "last_active": float(m.get("last_active", 0) or 0),
    }


def fetch_pending(r, sid: str, limit: int = 200):
    """取出"游标之后、还没压过"的消息，按时间正序返回。

    先用 XREVRANGE 取最近 limit 条再翻转，避免会话很长时把整条流拉进内存。
    """
    msgs = r.xrevrange(_key(sid, "msgs"), count=limit)
    msgs.reverse()

    last = get_meta(r, sid)["last_compacted_id"]
    if last and last != "0-0":
        # 不能用字符串比较 ID（"10-1" < "9-1"），所以按位置找
        idx = next((i for i, (mid, _) in enumerate(msgs) if mid == last), -1)
        if idx >= 0:
            msgs = msgs[idx + 1:]
    return msgs


def commit_compaction(r, sid: str, summary: str, facts: dict,
                      new_last_id: str, removed_ids: list,
                      retained_tokens: int = 0) -> None:
    """把压缩结果原子写回：摘要、事实、游标、删除已压消息。

    这四步必须放在一个事务里。否则中途失败会出现
    「消息已经 XDEL 了，但摘要没落盘」—— 信息永久丢失，
    这是整个方案里唯一真正危险的地方。
    """
    meta = _key(sid, "meta")
    ver = get_meta(r, sid)["summary_ver"]

    pipe = r.pipeline(transaction=True)
    pipe.set(_key(sid, "summary"), summary)
    if facts:
        pipe.hset(_key(sid, "facts"), mapping={str(k): str(v) for k, v in facts.items()})
    pipe.hset(meta, mapping={
        "last_compacted_id": new_last_id,
        "summary_ver": ver + 1,
        # token 重新计数：保留窗口里没被压的那几条 + 新摘要本身
        "tokens": retained_tokens + estimate_tokens(summary),
    })
    if removed_ids:
        pipe.xdel(_key(sid, "msgs"), *removed_ids)
    pipe.execute()


def load_for_context(r, sid: str, recent_count: int):
    """一次 pipeline 并发取齐：摘要、事实、最近 N 条原文。

    三个请求合并成一趟网络往返，比串行快。
    """
    pipe = r.pipeline(transaction=False)
    pipe.get(_key(sid, "summary"))
    pipe.hgetall(_key(sid, "facts"))
    pipe.xrevrange(_key(sid, "msgs"), count=recent_count)
    summary, facts, recent = pipe.execute()

    recent.reverse()
    return summary or "", facts or {}, recent


def clear_session(r, sid: str) -> None:
    """清空一个会话的所有 key（调试用）。"""
    for suffix in ("msgs", "meta", "summary", "facts", "lock"):
        r.delete(_key(sid, suffix))


def dump_stats(r, sid: str) -> dict:
    """汇总当前会话状态，给 /stats 指令和自检脚本用。"""
    meta = get_meta(r, sid)
    return {
        "msgs": r.xlen(_key(sid, "msgs")),
        "summary": r.get(_key(sid, "summary")) or "",
        "facts": r.hgetall(_key(sid, "facts")) or {},
        **meta,
    }
