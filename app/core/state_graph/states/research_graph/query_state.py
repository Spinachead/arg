from dataclasses import dataclass
from core.state_graph.states.main_graph.input_state import InputState


@dataclass(kw_only=True)
class QueryState(InputState):
    """State class for managing research queries in the research graph."""
    query: str = field(default="")
    sql: str = field(default="")
    sql_result: str = field(default="")
