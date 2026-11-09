"""
MySQL 连接与只读查询封装。

所有数据库访问都走这里：
  - run_query()       执行只读 SQL（只允许 SELECT / SHOW / DESCRIBE / EXPLAIN / WITH）
  - list_tables()     列出 employees 库里的表
  - get_table_schema() 查看某张表的建表语句 + 样例数据

连接信息从同目录的 .env 读取，代码里不写死密码。
"""

import os
import re
import time
from pathlib import Path

import pymysql
from dotenv import load_dotenv
from pymysql.cursors import DictCursor

# ========== 读取 .env，组装数据库连接配置 ==========
# 密码等敏感信息只放在 .env（已加入 .gitignore），代码里不写死
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "employees"),
    "charset": "utf8mb4",
}

# 单次查询最多返回多少行（防止大表撑爆模型上下文）和查询超时秒数
DEFAULT_MAX_ROWS = int(os.getenv("MAX_ROWS", "100"))
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "60"))

# ========== 只读 SQL 白名单（安全边界） ==========
# 1) 语句必须以这些关键字开头
_ALLOWED_START = ("select", "show", "describe", "desc", "explain", "with")

# 2) 任何位置出现这些关键字都拒绝执行，防止写操作
_FORBIDDEN_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|grant|revoke|"
    r"call|load|outfile|dumpfile|rename|lock|unlock|handler|prepare|execute|"
    r"deallocate|set|use|into|kill|shutdown)\b",
    re.IGNORECASE,
)

# 用于去掉注释后再做安全校验
_COMMENT_RE = re.compile(r"/\*.*?\*/|--[^\n]*|#[^\n]*", re.DOTALL)

# 字符串 / 反引号标识符，安全校验前先替换掉，避免把字符串内容当成关键字
_STRING_RE = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.|\"\")*\"|`[^`]*`")


class DatabaseError(Exception):
    """数据库相关错误（连接失败、SQL 被拒绝等），供工具层统一捕获。"""


# ========== 连接数据库 ==========
def get_connection():
    """新建一个 MySQL 连接（DictCursor，查询结果按列名返回）。"""
    try:
        return pymysql.connect(
            cursorclass=DictCursor,
            connect_timeout=5,
            read_timeout=QUERY_TIMEOUT,
            write_timeout=QUERY_TIMEOUT,
            **DB_CONFIG,
        )
    except pymysql.MySQLError as exc:
        raise DatabaseError(_friendly_connection_error(exc)) from exc


def _friendly_connection_error(exc):
    """把 MySQL 错误码翻译成用户能看懂的中文提示。"""
    code = exc.args[0] if exc.args else None
    if code == 1045:
        return (
            "数据库连接失败：用户名或密码不正确。请检查 .env 中的 "
            "DB_USER / DB_PASSWORD 是否与本地 MySQL 一致。"
        )
    if code == 1049:
        return (
            f"数据库 {DB_CONFIG['database']} 不存在。请先导入 employees.sql：\n"
            "  mysql -u root -p < employees.sql\n"
            "或运行 python check_db.py 查看导入方法。"
        )
    if code == 2003:
        return (
            f"无法连接 MySQL 服务 {DB_CONFIG['host']}:{DB_CONFIG['port']}，"
            "请确认 MySQL80 服务已启动。"
        )
    return f"数据库连接失败：{exc}"


# ========== 连接自检 ==========
def check_connection():
    """检查连接是否正常，返回 (ok, message)。"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT VERSION() AS version")
                version = cur.fetchone()["version"]
        return True, f"已连接 MySQL {version}，数据库：{DB_CONFIG['database']}"
    except DatabaseError as exc:
        return False, str(exc)
    except pymysql.MySQLError as exc:
        return False, f"数据库连接失败：{exc}"


# ========== 安全校验：只允许单条只读语句 ==========
def validate_readonly_sql(sql: str) -> str:
    """
    校验 SQL 是否为单条只读语句，通过则返回清理后的 SQL，否则抛 DatabaseError。
    """
    if not sql or not sql.strip():
        raise DatabaseError("SQL 为空。")

    raw = sql.strip()
    # 校验时忽略注释和字符串内容，执行时仍使用原 SQL
    cleaned = _COMMENT_RE.sub(" ", raw).strip()
    cleaned = _STRING_RE.sub("''", cleaned)

    # 去掉结尾分号，再检查是否还有分号（防止多语句）
    cleaned = cleaned.rstrip(";").strip()
    if ";" in cleaned:
        raise DatabaseError("只允许执行一条 SQL 语句，请去掉多余的分号。")

    first_word = cleaned.split(None, 1)[0].lower() if cleaned.split() else ""
    if first_word not in _ALLOWED_START:
        raise DatabaseError(
            "只能执行只读查询（SELECT / SHOW / DESCRIBE / EXPLAIN / WITH），"
            f"当前语句以 {first_word!r} 开头，已被拒绝。"
        )

    match = _FORBIDDEN_RE.search(cleaned)
    if match:
        raise DatabaseError(
            f"SQL 中包含不允许的关键字 {match.group(0)!r}，只能执行只读查询。"
        )

    return raw


# ========== 执行只读查询 ==========
def run_query(sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    """
    执行只读 SQL。

    返回:
        {
          "columns": [列名...],
          "rows": [ {列名: 值}, ... ],
          "row_count": 返回行数,
          "truncated": 是否因 max_rows 被截断,
          "elapsed_ms": 耗时,
        }
    """
    safe_sql = validate_readonly_sql(sql)
    max_rows = max(1, int(max_rows))

    start = time.perf_counter()
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(safe_sql)
                # 多取 1 行，用来判断结果是否被截断
                rows = cur.fetchmany(max_rows + 1)
                columns = [d[0] for d in cur.description] if cur.description else []
    except DatabaseError:
        raise
    except pymysql.MySQLError as exc:
        raise DatabaseError(f"SQL 执行失败：{exc}") from exc

    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "elapsed_ms": elapsed_ms,
    }


# ========== 表结构相关查询 ==========
def list_tables() -> list[dict]:
    """返回数据库中的所有表及大致行数。"""
    sql = (
        "SELECT TABLE_NAME AS table_name, TABLE_ROWS AS table_rows "
        "FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = %s ORDER BY TABLE_NAME"
    )
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (DB_CONFIG["database"],))
                return list(cur.fetchall())
    except DatabaseError:
        raise
    except pymysql.MySQLError as exc:
        raise DatabaseError(f"读取表列表失败：{exc}") from exc


def get_table_schema(table_name: str) -> dict:
    """查看表的建表语句和 3 行样例数据。表名会做白名单校验。"""
    if not re.fullmatch(r"[A-Za-z0-9_]+", table_name or ""):
        raise DatabaseError(f"非法表名：{table_name!r}")

    existing = {t["table_name"] for t in list_tables()}
    if table_name not in existing:
        raise DatabaseError(
            f"表 {table_name!r} 不存在，可用的表：{', '.join(sorted(existing))}"
        )

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SHOW CREATE TABLE `{table_name}`")
                create_row = cur.fetchone()
                create_sql = next(
                    (v for k, v in create_row.items() if "create" in k.lower()), ""
                )
                cur.execute(f"SELECT * FROM `{table_name}` LIMIT 3")
                samples = list(cur.fetchall())
    except DatabaseError:
        raise
    except pymysql.MySQLError as exc:
        raise DatabaseError(f"读取表结构失败：{exc}") from exc

    return {"table": table_name, "create_sql": create_sql, "samples": samples}


# ========== 查询结果格式化（转成模型易读的文本表格） ==========
def format_rows(rows: list[dict], columns: list[str], max_display: int = 30) -> str:
    """把查询结果格式化成易于 LLM 阅读的文本表格。"""
    if not rows:
        return "(查询结果为空)"

    display = rows[:max_display]
    lines = [" | ".join(columns), "-" * 40]
    for row in display:
        lines.append(" | ".join(_cell(row.get(c)) for c in columns))
    if len(rows) > max_display:
        lines.append(f"... 其余 {len(rows) - max_display} 行未展示")
    return "\n".join(lines)


def _cell(value) -> str:
    if value is None:
        return "NULL"
    text = str(value)
    return text.replace("\n", " ").replace("|", "/")
