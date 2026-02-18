from dataclasses import dataclass, field
from typing import Annotated, List, Dict
from core.state_graph.states.main_graph.input_state import InputState
from core.state_graph.states.main_graph.router import Router

@dataclass(kw_only=True)
class AgentState(InputState):
    """
    表示agent在主状态图中的状态。
    """
    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
    context: str = field(default="")