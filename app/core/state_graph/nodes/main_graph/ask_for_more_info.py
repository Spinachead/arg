from core.state_graph.states.main_graph.agent_state import AgentState
from core.prompts import MORE_INFO_SYSTEM_PROMPT
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from settings import Settings


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """
    根据当前的route logic向用户询问更多信息。
    """

    model = init_chat_model(
        name="ask_for_more_info",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )
    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(logic=state.router.logic)
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages, config)
    return {"messages": [response]}
