from typing import TypedDict
from typing import Literal


class Step(TypedDict):
    """Single research step"""
    question: str
    type: Literal["semantic_search", "query_search"]
