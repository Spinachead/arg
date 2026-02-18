from dataclasses import dataclass, field
from core.state_graph.states.main_graph.input_state import InputState

@dataclass(kw_only=True)
class ResearcherState(InputState):
    """State of the researcher graph."""
    queries: list[str] = field(default_factory=list)
    context: str = field(default="")
