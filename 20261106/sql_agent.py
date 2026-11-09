"""
简单的 SQL 查询智能体（LangChain + 本地 Ollama + MySQL employees 库）。

用法：
    D:/conda_envs/langchain/python.exe sql_agent.py              # 交互式提问
    D:/conda_envs/langchain/python.exe sql_agent.py "有多少员工？"  # 单次提问
    ... -v                                                       # 显示生成的 SQL

流程：生成 SQL → 逻辑检查 → 执行（失败自动修正，最多 3 次）→ 根据真实结果回答。
对本地 Ollama 模型（qwen2.5-coder:7b / qwen2.5:1.5b）都适用。
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import db

# ========== 读取 .env 配置 ==========
# .env 与代码放在同一目录；用 Path(__file__) 定位，从任何目录运行都能找到
load_dotenv(Path(__file__).with_name(".env"))

# Ollama 的 OpenAI 兼容接口地址；模型名、占位 API Key（Ollama 不校验）、温度

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# ========== 数据库表结构（作为提示词交给模型，让它知道有哪些表和字段） ==========
SCHEMA_TEXT = """employees(emp_no, birth_date, first_name, last_name, gender, hire_date)  员工
departments(dept_no, dept_name)  部门
dept_emp(emp_no, dept_no, from_date, to_date)  员工任职部门
dept_manager(dept_no, emp_no, from_date, to_date)  部门经理
salaries(emp_no, salary, from_date, to_date)  薪资（年薪）
titles(emp_no, title, from_date, to_date)  职称"""

# ========== 生成 SQL 的规则（针对 employees 库的常见错误总结） ==========
SQL_RULES = """规则：
1. 只能写一条只读查询（SELECT / WITH），禁止任何写操作。
2. 只有 dept_emp、dept_manager、salaries、titles 这四张表有 from_date / to_date；employees 和 departments 没有 to_date，不要给它们加这个条件。
3. to_date = '9999-01-01' 表示当前有效。问“现在/当前”时，涉及的上述表都要加该条件（当前工资要加 s.to_date，当前部门要加 de.to_date）。
4. 工资在 salaries 表，不在 employees 表；查工资要 JOIN salaries s ON s.emp_no = e.emp_no。
5. 部门名称在 departments 表，通过 dept_emp 关联。
6. 问题里的筛选条件都要写进 SQL：女性 = gender='F'，男性 = gender='M'，年份用 hire_date / from_date / to_date 比较。
7. 问“有多少人 / 几个”时，只返回 COUNT(*)，不要 SELECT 员工明细。
8. GROUP BY 的列必须同时出现在 SELECT 里，否则无法区分每组数据。
9. 只 JOIN 问题真正需要的表，不要加入无关的表；每张表必须使用不同的别名。
10. 只查询需要的列；结果可能很多时加 LIMIT 50。
11. 今天是 2026-11-09。"""

# ========== 提示词 1：把用户问题翻译成 SQL ==========
SQL_SYSTEM_PROMPT = f"""你是 MySQL 查询生成器。把用户问题翻译成一条只读 SQL。
只输出 SQL 本身，不要解释，不要 markdown 代码块，不要结尾分号。

数据库 employees 的表结构：
{SCHEMA_TEXT}

{SQL_RULES}

示例 1：
问题：当前工资最高的 5 名员工是谁？
SELECT e.first_name, e.last_name, s.salary FROM employees e JOIN salaries s ON s.emp_no = e.emp_no WHERE s.to_date = '9999-01-01' ORDER BY s.salary DESC LIMIT 5

示例 2：
问题：当前男女员工各有多少人？
SELECT gender, COUNT(*) AS total FROM employees GROUP BY gender

示例 3：
问题：女性员工里当前工资超过 10 万美元的有多少人？
SELECT COUNT(*) AS total FROM employees e JOIN salaries s ON s.emp_no = e.emp_no WHERE e.gender = 'F' AND s.to_date = '9999-01-01' AND s.salary > 100000

示例 4：
问题：现在哪个部门的平均工资最高？
SELECT d.dept_name, AVG(s.salary) AS avg_salary FROM departments d JOIN dept_emp de ON de.dept_no = d.dept_no JOIN salaries s ON s.emp_no = de.emp_no WHERE de.to_date = '9999-01-01' AND s.to_date = '9999-01-01' GROUP BY d.dept_name ORDER BY avg_salary DESC LIMIT 1

示例 5：
问题：当前每个职称有多少人？
SELECT title, COUNT(*) AS total FROM titles WHERE to_date = '9999-01-01' GROUP BY title ORDER BY total DESC

示例 6：
问题：1990 年之前入职的员工现在平均工资是多少？
SELECT AVG(s.salary) AS avg_salary FROM employees e JOIN salaries s ON s.emp_no = e.emp_no WHERE e.hire_date < '1990-01-01' AND s.to_date = '9999-01-01'

错误示例（employees 表没有 to_date，这样写会报错）：
SELECT COUNT(*) FROM employees WHERE to_date = '9999-01-01'"""

# ========== 提示词 2：SQL 执行失败或逻辑检查不通过时，带错误信息让模型修复 ==========
REPAIR_SYSTEM_PROMPT = f"""你是 MySQL 专家。下面这条 SQL 在 employees 数据库执行失败或逻辑有误，请修正它。
不要原样重复有错误的 SQL，必须根据错误信息修改。只输出修正后的一条 SQL，不要解释，不要 markdown 代码块，不要结尾分号。

数据库 employees 的表结构：
{SCHEMA_TEXT}

{SQL_RULES}"""

# ========== 提示词 3：根据真实查询结果用中文回答（禁止编造数据） ==========
ANSWER_SYSTEM_PROMPT = """你是数据分析助手，根据 SQL 查询结果回答用户问题。
要求：
- 用中文简洁回答，先给结论，再补充必要的细节。
- 只能使用查询结果里的数字，绝对不要编造数据。
- 如果结果为空，如实说明没有查询到数据。
- 如果结果被截断，说明答案只基于前面的部分数据。
- 字段值要转成中文：gender 的 M = 男，F = 女。
- 可以简单解释查询思路，但除非用户要求，不要输出完整 SQL。"""

# ========== 问候 / 帮助类问题的快捷判断 ==========
# 这类问题不需要查数据库，直接用正则拦下，比让模型生成 SQL 更稳、更快
_GREETING_RE = re.compile(
    r"^\s*(你好|您好|hi|hello|嗨|在吗|谢谢|感谢|再见)[\s!！。.,，]*$", re.IGNORECASE
)
_HELP_RE = re.compile(r"你能做什么|能干什么|帮助|help|怎么用", re.IGNORECASE)

HELP_TEXT = """我可以把自然语言问题翻译成 SQL，查询本地 employees 数据库并回答。
可以问：
- 一共有多少名员工？
- 当前工资最高的 5 名员工是谁？
- 现在哪个部门的平均工资最高？
- 当前男女员工各有多少人？
- 1990 年之前入职的员工现在平均工资是多少？"""


# ========== 创建 LLM 客户端 ==========
def build_llm(temperature: float | None = None) -> ChatOpenAI:
    """OpenAI 兼容接口调用本地 Ollama。"""
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
    )


# ========== LangChain 管线：提示词 → 模型 → 纯文本 ==========
# 每条管线 = ChatPromptTemplate + LLM + StrOutputParser，
# 调用 invoke({变量字典}) 就能得到模型输出的纯文本。

def build_sql_chain(llm):
    """生成 SQL 的管线。"""
    prompt = ChatPromptTemplate.from_messages(
        [("system", SQL_SYSTEM_PROMPT), ("human", "{input}")]
    )
    return prompt | llm | StrOutputParser()


def build_repair_chain(llm):
    """修复 SQL 的管线：把“问题 + 失败 SQL + 错误信息”一起交给模型改写。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPAIR_SYSTEM_PROMPT),
            (
                "human",
                "用户问题：{question}\n\n执行失败的 SQL：{sql}\n\n错误信息：{error}\n\n"
                "请输出修正后的 SQL。",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_answer_chain(llm):
    """组织回答的管线：把真实查询结果交给模型，用中文总结。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            (
                "human",
                "用户问题：{question}\n\n执行的 SQL：{sql}\n\n"
                "查询结果（共 {row_count} 行{extra}）：\n{result_text}",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


# ========== SQL 清洗：从模型输出里提取一条干净的 SQL ==========
def clean_sql(text: str) -> str:
    """去掉 markdown 代码块、sql: 前缀、结尾分号等模型爱加的包装。"""
    text = text.strip()
    # 去掉 ```sql ... ``` 代码块围栏
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    # 去掉 “sql：” 这样的前缀
    text = re.sub(r"^\s*sql\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    # 如果前面还有解释性文字，从第一个 select/with/... 开始截取
    match = re.search(r"(?is)\b(select|with|show|describe|explain)\b.*", text)
    if match:
        text = match.group(0)
    # 只保留分号前的第一条语句
    return text.split(";")[0].strip()


# ========== SQL 逻辑检查（不连数据库，先用正则拦下常见错误） ==========
# FROM / JOIN 后面紧跟的可能是这些关键字而不是别名，提取别名时要排除
_ALIAS_STOPWORDS = {
    "where", "join", "inner", "left", "right", "full", "cross", "on",
    "using", "group", "order", "having", "limit", "union", "offset",
    "natural", "straight_join", "as",
}


def _table_aliases(sql: str) -> list[tuple[str, str]]:
    """粗略提取 FROM / JOIN 后面的表名和别名。"""
    aliases = []
    pattern = re.compile(
        r"(?is)\b(?:from|join)\s+`?(\w+)`?(?:\s+(?:as\s+)?`?(\w+)`?)?"
    )
    for match in pattern.finditer(sql):
        table = match.group(1).lower()
        alias = match.group(2)
        if alias and alias.lower() in _ALIAS_STOPWORDS:
            alias = None
        aliases.append((table, (alias or table).lower()))
    return aliases


def sql_sanity_issues(sql: str) -> list[str]:
    """
    轻量逻辑检查：在执行前拦下几类常见错误，
    把问题描述交给修复环节，让模型针对性改写。
    """
    issues = []
    aliases = _table_aliases(sql)

    # 1) 表别名重复（例如 JOIN 时两次用了 de）
    seen_aliases = set()
    for _, alias in aliases:
        if alias in seen_aliases:
            issues.append(f"表别名 {alias} 被重复使用，每张表要使用不同别名")
        seen_aliases.add(alias)

    # 2) 自连接的恒等式（例如 ON de.dept_no = de.dept_no）
    for match in re.finditer(
        r"(?is)\bon\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", sql
    ):
        if (
            match.group(1).lower() == match.group(3).lower()
            and match.group(2).lower() == match.group(4).lower()
        ):
            issues.append(f"JOIN 条件 {match.group(0)[3:]} 是自己等于自己，没有意义")

    # 3) 给 employees 表加了不存在的 to_date 字段
    for table, alias in aliases:
        if table == "employees" and re.search(
            rf"(?i)\b{re.escape(alias)}\.to_date\b", sql
        ):
            issues.append(
                f"{alias} 是 employees 表的别名，employees 没有 to_date 字段"
            )
    tables = {table for table, _ in aliases}
    if "to_date" in sql.lower() and not (
        tables & {"dept_emp", "dept_manager", "salaries", "titles"}
    ):
        issues.append("查询没有涉及带 to_date 的表，不能使用 to_date 条件")

    # 4) GROUP BY 的列没有出现在 SELECT 中
    group_match = re.search(
        r"(?is)\bgroup\s+by\s+(.+?)(?:\bhaving\b|\border\s+by\b|\blimit\b|$)",
        sql,
    )
    if group_match:
        if sql.lstrip().lower().startswith("with"):
            # CTE 时取最后一个 SELECT ... FROM（主查询）
            select_matches = re.findall(r"(?is)\bselect\s+(.+?)\s+from\b", sql)
            select_text = select_matches[-1] if select_matches else ""
        else:
            select_match = re.search(r"(?is)^\s*select\s+(.+?)\s+from\b", sql)
            select_text = select_match.group(1) if select_match else ""
        for column in group_match.group(1).split(","):
            column = column.strip()
            if not column:
                continue
            base = column.split(".")[-1].strip("`")
            if base not in select_text:
                issues.append(f"GROUP BY 的列 {column} 没有出现在 SELECT 中")

    return issues


# ========== 历史对话格式化 ==========
# 只取最近 6 条拼进提示词，让模型能理解“那个部门呢？”这类追问
def format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = [f"{item['role']}：{item['content']}" for item in history[-6:]]
    return "历史对话（供理解指代）：\n" + "\n".join(lines) + "\n\n"


# ========== 核心流程：问题 → 生成 SQL → 检查/执行 → 失败修复 → 组织回答 ==========
def ask_pipeline(llm, question: str, history: list[dict], verbose: bool = False) -> tuple[str, str | None]:
    """返回 (回答, 最后执行的 SQL)。"""
    if _GREETING_RE.match(question):
        return "你好！我是 employees 数据库的查询助手，可以问我员工、部门、薪资、职称相关的问题。", None
    if _HELP_RE.search(question):
        return HELP_TEXT, None

    input_text = format_history(history) + f"用户问题：{question}"
    sql = clean_sql(build_sql_chain(llm).invoke({"input": input_text}))

    result = None
    last_error = None
    if not sql:
        last_error = "模型没有生成 SQL。"
    else:
        # 最多尝试 3 轮：逻辑检查 → 执行 → 失败则让模型修正后再试
        for attempt in range(3):
            if verbose:
                print(f"  [SQL {attempt + 1}] {sql}")

            # 先做轻量逻辑检查，能省一次数据库往返
            issues = sql_sanity_issues(sql)
            if issues:
                last_error = "SQL 逻辑检查未通过：" + "；".join(issues)
                if verbose:
                    print(f"  [检查] {last_error}")
            else:
                try:
                    result = db.run_query(sql)
                    break
                except db.DatabaseError as exc:
                    last_error = str(exc)
                    if verbose:
                        print(f"  [失败] {last_error}")

            if attempt == 2:  # 最后一次也失败，不再修复
                break
            # 把失败原因反馈给模型，让它针对性改写
            repaired = build_repair_chain(llm).invoke(
                {"question": question, "sql": sql, "error": last_error}
            )
            sql = clean_sql(repaired)
            if not sql:
                last_error = "模型没有生成有效的 SQL。"
                break

    if result is None:
        answer = f"查询失败：{last_error}\n最后一次的 SQL：{sql or '(空)'}"
        history.extend([
            {"role": "用户", "content": question},
            {"role": "助手", "content": answer},
        ])
        return answer, sql

    extra = "，已截断" if result["truncated"] else ""
    result_text = db.format_rows(result["rows"], result["columns"], max_display=30)
    if verbose:
        print(f"  [结果] {result['row_count']} 行，耗时 {result['elapsed_ms']} ms")

    answer = build_answer_chain(llm).invoke(
        {
            "question": question,
            "sql": sql,
            "row_count": result["row_count"],
            "extra": extra,
            "result_text": result_text,
        }
    ).strip()

    history.extend(
        [
            {"role": "用户", "content": question},
            {"role": "查询", "content": sql},
            {"role": "助手", "content": answer},
        ]
    )
    return answer, sql


# ========== 命令行入口 ==========
# 不带参数 → 交互模式；带问题参数 → 单次提问后退出；-v 显示 SQL 和执行过程
def main():
    import argparse

    parser = argparse.ArgumentParser(description="employees 数据库 SQL 查询智能体")
    parser.add_argument("question", nargs="*", help="要问的问题；不填则进入交互模式")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示生成的 SQL 和执行过程")
    args = parser.parse_args()

    ok, message = db.check_connection()
    if not ok:
        print(f"[数据库] {message}")
        return 1

    llm = build_llm()
    print(f"[数据库] {message}")
    print(f"[模型] {LLM_MODEL}  ({LLM_BASE_URL})")
    print("输入问题开始查询，输入 exit 退出。\n")

    history: list[dict] = []

    def answer(question: str) -> str:
        text, _ = ask_pipeline(llm, question, history, args.verbose)
        return text

    if args.question:
        print(answer(" ".join(args.question)))
        return 0

    while True:
        try:
            question = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("再见！")
            break

        try:
            print(f"助手> {answer(question)}\n")
        except Exception as exc:  # noqa: BLE001 - CLI 里给出可读错误即可
            print(f"[出错] {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
