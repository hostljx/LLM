"""
数据库自检脚本：确认 .env 配置、MySQL 连接、employees 库和表是否就绪。

用法：
    D:/conda_envs/langchain/python.exe check_db.py
    D:/conda_envs/langchain/python.exe check_db.py --import-dump   # 库不存在时导入 employees.sql
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import db

# employees.sql 的路径：与脚本同目录
DUMP_FILE = Path(__file__).with_name("employees.sql")

# ========== 定位 mysql 客户端 ==========
# 先看 PATH 里有没有 mysql，再依次检查这些常见安装路径
MYSQL_CANDIDATES = [
    r"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe",
    r"C:\Program Files\MySQL\MySQL Server 9.0\bin\mysql.exe",
    r"C:\Program Files\MySQL\MySQL Server 8.4\bin\mysql.exe",
]


def find_mysql() -> str | None:
    found = shutil.which("mysql")
    if found:
        return found
    for path in MYSQL_CANDIDATES:
        if Path(path).exists():
            return path
    return None


# ========== 导入 employees.sql ==========
def import_dump() -> int:
    """用 mysql 客户端导入 employees.sql（库不存在时使用）。"""
    mysql = find_mysql()
    if not mysql:
        print("找不到 mysql 客户端，请手动导入：")
        print("  mysql -u root -p < employees.sql")
        return 1
    if not DUMP_FILE.exists():
        print(f"找不到数据文件：{DUMP_FILE}")
        return 1

    env = os.environ.copy()
    env["MYSQL_PWD"] = db.DB_CONFIG["password"]
    base_cmd = [
        mysql,
        f"-h{db.DB_CONFIG['host']}",
        f"-P{db.DB_CONFIG['port']}",
        f"-u{db.DB_CONFIG['user']}",
    ]

    # dump 里没有 CREATE DATABASE / USE，需要先建库并指定默认库
    print(f"创建数据库 {db.DB_CONFIG['database']}（如不存在）...")
    create = subprocess.run(
        base_cmd + ["-e", f"CREATE DATABASE IF NOT EXISTS `{db.DB_CONFIG['database']}`"],
        env=env,
    )
    if create.returncode != 0:
        print("创建数据库失败，请检查 .env 里的账号密码。")
        return create.returncode

    print(f"开始导入 {DUMP_FILE.name}（305MB，可能需要几分钟）...")
    cmd = base_cmd + [f"-D{db.DB_CONFIG['database']}"]
    with DUMP_FILE.open("rb") as stdin:
        proc = subprocess.run(cmd, stdin=stdin, env=env)
    if proc.returncode == 0:
        print("导入完成。")
    else:
        print(f"导入失败，退出码 {proc.returncode}。")
    return proc.returncode


# ========== 自检主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="检查 employees 数据库连接")
    parser.add_argument(
        "--import-dump",
        action="store_true",
        help="数据库不存在时导入同目录下的 employees.sql",
    )
    args = parser.parse_args()

    print(f"配置：{db.DB_CONFIG['user']}@{db.DB_CONFIG['host']}:"
          f"{db.DB_CONFIG['port']}/{db.DB_CONFIG['database']}")

    # 1) 先测试能否连上数据库
    ok, message = db.check_connection()
    print(message)

    if not ok:
        if args.import_dump:
            return import_dump()
        if "不存在" in message and DUMP_FILE.exists():
            print("\n可以运行以下命令导入数据：")
            print(f"  python check_db.py --import-dump")
        return 1

    # 2) 列出所有表，确认 employees.sql 已导入
    tables = db.list_tables()
    print(f"\n共 {len(tables)} 张表：")
    for t in tables:
        print(f"  - {t['table_name']}: 约 {t['table_rows']} 行")

    # 3) 实际执行一条查询，确认权限和表结构都正常
    print("\n试查 employees 表前 3 行：")
    result = db.run_query("SELECT emp_no, first_name, last_name, hire_date "
                          "FROM employees ORDER BY emp_no LIMIT 3")
    print(db.format_rows(result["rows"], result["columns"]))
    print("\n数据库可以正常使用 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
