import pymysql
from langchain_core.tools import tool
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()


@tool
def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    """安全执行只读 SELECT 查询。禁止 INSERT/UPDATE/DELETE/DROP。"""

    # 1. 安全校验：只允许 SELECT，且不能有危险关键词
    query = query.strip().lower()
    if not query.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    dangerous_keywords = ["drop", "delete", "update", "insert", "exec", "union", "--"]
    if any(kw in query for kw in dangerous_keywords):
        raise ValueError("Potentially dangerous query detected.")
    try:
        # 从环境变量读取 MySQL 配置
        host = os.getenv("MYSQL_DB_HOST", "localhost")
        # 如果 host 包含端口，需要分离
        if ":" in host:
            host, port_str = host.split(":")
            port = int(port_str)
        else:
            port = int(os.getenv("MYSQL_DB_PORT", "3306"))
        
        connection = pymysql.connect(
            host=host,
            port=port,
            user=os.getenv("MYSQL_DB_USERNAME", "root"),
            password=os.getenv("MYSQL_DB_PASSWORD", ""),
            database=os.getenv("MYSQL_DB_DATABASE", "arg")
        )
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
        connection.close()
        return result
    except Exception as e:
        raise ValueError(f"Error executing query: {e}")