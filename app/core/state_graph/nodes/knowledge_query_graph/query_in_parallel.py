from core.state_graph.states.knowledge_query_graph.researcher_state import ResearcherState
from core.state_graph.states.knowledge_query_graph.query_state import QueryState

from langgraph.types import Send


def query_in_parallel(state: ResearcherState) -> list[Send]:
    """
    Create parallel Send messages to execute multiple queries concurrently.
    
    This function takes a ResearcherState containing multiple queries and creates
    a list of Send messages that will dispatch each query to the "execute_query"
    node for parallel execution. This enables concurrent processing of multiple
    database queries to improve performance.
    
    Args:
        state (ResearcherState): The current researcher state containing a list
                                of queries to be executed in parallel
    
    Returns:
        list[Send]: A list of Send messages, each configured to execute one query
                   in the "execute_query" node with a QueryState containing the
                   individual query
    """
    return [Send("execute_query", QueryState(query=query)) for query in state.queries]