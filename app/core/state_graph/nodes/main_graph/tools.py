from langchain_core.tools import tool
import pymysql  # 或 psycopg2（PostgreSQL）

@tool
def get_sn_table_count() -> int:
    """Get the total number of records in the 'sn' table."""
    try:
        # MySQL 示例
        connection = pymysql.connect(
            host="localhost",
            user="arg",
            password="12345678",
            database="arg"
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM sn")
            count = cursor.fetchone()[0]
        connection.close()
        return count
    except Exception as e:
        print(f"Error querying sn table: {str(e)}")
        return f"Error querying sn table: {str(e)}"



# 在这里集中管理所有普通工具
# 添加新工具时，只需：
# 1. 在上面定义工具函数（使用 @tool 装饰器）
# 2. 将工具添加到下面的列表中
GENERAL_TOOLS = [
    get_sn_table_count,
    # 在这里添加更多工具...
    # new_tool_1,
    # new_tool_2,
]