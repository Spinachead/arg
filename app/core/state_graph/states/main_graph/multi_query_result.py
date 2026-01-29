from pydantic import BaseModel, Field
from typing import List


class QueryWithKB(BaseModel):
    """查询变体及其对应的知识库"""
    query: str = Field(description="重写后的查询文本")
    kb_name: str = Field(description="最适合该查询的知识库名称")

class MultiQueryResult(BaseModel):
    """多个查询变体的结果"""
    queries: List[QueryWithKB] = Field(description="查询变体列表，包含查询文本和对应的知识库名称")