from core.state_graph.states.main_graph.agent_state import AgentState
from typing import Literal


def route_query(
    state: AgentState,
) -> Literal["conduct_knowledge", "ask_for_more_info", "respond_to_general_query"]:
    """
    Determines the next action for the agent based on the router type in the current state.

    Args:
        state (AgentState): The current state of the agent, including the router type.

    Returns:
        Literal["conduct_knowledge", "ask_for_more_info", "respond_to_general_query"]:
            The next node/action to execute in the state graph.

    Raises:
        ValueError: If the router type is unknown.
    """
    _type = state.router.type
    if _type == "valid_knowledge_base":
        return "conduct_knowledge"
    elif _type == "valid_sql_query":
        return "conduct_knowledge"
    elif _type == "valid_log_query":
        return "conduct_knowledge"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")
