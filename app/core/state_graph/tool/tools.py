from langchain_core.tools import tool
import pymysql
import os
from dotenv import load_dotenv
load_dotenv()

@tool
def get_sn_table_count() -> int:
    """Get the total number of records in the 'sn' table."""
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
            cursor.execute("SELECT COUNT(*) FROM sn")
            count = cursor.fetchone()[0]
        connection.close()
        return count
    except Exception as e:
        print(f"Error querying sn table: {str(e)}")
        return f"Error querying sn table: {str(e)}"


@tool
def execute_sql_query(sql: str) -> str:
    """执行sql查询语句并且返回结果.

    Args:
        sql: The SQL query string to execute
    """
    try:
        host = os.getenv("MYSQL_DB_HOST", "localhost")
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
            cursor.execute(sql)
            # 如果是 SELECT 查询，获取结果
            if sql.strip().upper().startswith("SELECT"):
                results = cursor.fetchall()
                connection.close()
                return str(results)
            else:
                connection.commit()
                connection.close()
                return "Query executed successfully"
    except Exception as e:
        return f"Error executing SQL: {str(e)}"



# 在这里集中管理所有普通工具
# 添加新工具时，只需：
# 1. 在上面定义工具函数（使用 @tool 装饰器）
# 2. 将工具添加到下面的列表中
GENERAL_TOOLS = [
    get_sn_table_count,
    execute_sql_query,
    # 在这里添加更多工具...
    # new_tool_1,
    # new_tool_2,
]