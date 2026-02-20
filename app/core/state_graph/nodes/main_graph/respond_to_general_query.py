from core.state_graph.states.main_graph.agent_state import AgentState
from core.prompts import GENERAL_SYSTEM_PROMPT
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from settings import Settings


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """
    Generates a response to a general user query based on the agent's current state and routing logic.
    """
    model = init_chat_model(
        name="respond_to_general_query",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )
    system_prompt = GENERAL_SYSTEM_PROMPT.format(logic=state.router.logic)
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages, config)
    print(f"\033[92mUsing respond_to_general_query: {response}\033[0m")  # 绿色输出
    return {"messages": [response]}
