from core.state_graph.states.main_graph.agent_state import AgentState
from langchain_core.runnables import RunnableConfig
from core.state_graph.states.main_graph.router import Router
from langchain.chat_models import init_chat_model
from settings import Settings
from core.prompts import ROUTER_SYSTEM_PROMPT
from typing import cast



async def analyze_and_route_query(state: AgentState, *, config: RunnableConfig) -> dict[str, Router]:
    """
    分析当前代理状态并确定下一步的route logic
    """

    model = init_chat_model(
        name="analyze_and_route_query",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )

    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}] + state.messages
    print("---ANALYZE AND ROUTE QUERY---")
    print(f"MESSAGES: {state.messages}")
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    return {"router": response}






