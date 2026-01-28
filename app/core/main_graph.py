from token import STAR
from langgraph.graph import END, START, StateGraph
from core.state_graph.states.main_graph.agent_state import AgentState
from core.state_graph.states.main_graph.input_state import InputState
from core.state_graph.nodes.main_graph.test import testRes



def build_main_graph():
    builder = StateGraph(AgentState, input=InputState)
    builder.add_node("test", testRes)
    
    return builder.compile()
