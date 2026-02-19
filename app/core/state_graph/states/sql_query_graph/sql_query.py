from dataclasses import dataclass, field
from typing import Annotated
from core.state_graph.states.main_graph.input_state import InputState


@dataclass(kw_only=True)
class SQLQueryState(InputState):
    """State of the SQLQuery graph."""
    sql: str = field(default="")
    context: str = field(default="")

