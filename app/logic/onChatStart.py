import chainlit as cl
from core.main_graph import build_main_graph
from core.state_graph.states.main_graph.input_state import InputState
from db.repository.message_repository import list_messages_by_conversation
from langchain_core.messages import HumanMessage, AIMessage


async def execute():
    cl.user_session.set("graph", build_main_graph())
    cl.user_session.set("state", InputState(messages=[]))

