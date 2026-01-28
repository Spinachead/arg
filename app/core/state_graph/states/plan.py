from typing import TypedDict
from core.state_graph.states.step import Step


class Plan(TypedDict):
    """Research plan in the meal planner system."""
    steps: list[Step]
