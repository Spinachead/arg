import chainlit as cl
from core.main_graph import build_main_graph
from core.state_graph.states.main_graph.input_state import InputState


async def execute():
    cl.user_session.set("graph", build_main_graph())
    cl.user_session.set("state", InputState(messages=[]))
