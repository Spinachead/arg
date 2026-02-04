from pydantic import BaseModel, Field
from typing import List


class QueryWithKB(BaseModel):
    """查询变体及其对应的知识库"""
    query: str = Field(description="重写后的查询文本")

class MultiQueryResult(BaseModel):
    """多个查询变体的结果"""
    queries: List[QueryWithKB] = Field(description="查询变体列表")