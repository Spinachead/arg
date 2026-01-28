from dataclasses import dataclass, field
from typing import Annotated
from core.state_graph.states.step import Step
from utils.utils import update_knowledge


@dataclass(kw_only=True)
class ResearcherState:
    """State of the researcher graph."""
    step: Step
    queries: list[str] = field(default_factory=list)
    knowledge: Annotated[list[dict], update_knowledge] = field(default_factory=list)
