from dataclasses import dataclass, field
from typing import Annotated
from core.state_graph.states.main_graph.input_state import InputState
from core.state_graph.states.main_graph.router import Router


@dataclass(kw_only=True)
class AgentState(InputState):
    """
    Represents the state of an agent within the main state graph.

    Attributes:
        router (Router): The routing logic for the agent.
        steps (list[Step]): The sequence of steps taken by the agent.
        knowledge (list[dict]): The agent's accumulated knowledge, updated via the update_knowledge function.
    """
    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
