from core.state_graph.states.main_graph.agent_state import AgentState
from typing import Literal


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """
    Determines the next action for the agent based on the router type in the current state.

    Args:
        state (AgentState): The current state of the agent, including the router type.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
            The next node/action to execute in the state graph.

    Raises:
        ValueError: If the router type is unknown.
    """
    _type = state.router.type
    if _type == "valid":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")
