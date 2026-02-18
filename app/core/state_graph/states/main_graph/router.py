from typing import Literal
from pydantic import BaseModel


class Router(BaseModel):
    logic: str
    type: Literal["more-info", "valid_knowledge_base", "valid_sql_query", "valid_log_query", "general"]
