from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from config import config as app_config
from core.state_graph.states.main_graph.agent_state import AgentState

async def testRes (
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    model = init_chat_model(name="respond", **app_config["inference_model_params"])
    response = await model.ainvoke(state.messages)
    return {"messages": [response]}
