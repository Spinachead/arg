from pydantic import BaseModel, Field
from typing import List

class MultiQueryResult(BaseModel):
    queries: List[str] = Field(description="查询变体列表")
