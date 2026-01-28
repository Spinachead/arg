from typing import Literal
from pydantic import BaseModel


class Router(BaseModel):
    logic: str
    type: Literal["more-info", "valid", "general"]
