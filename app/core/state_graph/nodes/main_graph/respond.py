from core.state_graph.states.main_graph.agent_state import AgentState
from core.prompts import RESPONSE_SYSTEM_PROMPT
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from settings import Settings



async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """
    Generates a final response to the user based on the agent's accumulated knowledge and messages.

    Args:
        state (AgentState): The current state of the agent, including knowledge and messages.
        config (RunnableConfig): Configuration for the runnable execution.

    Returns:
        dict[str, list[BaseMessage]]: A dictionary containing the generated response message(s).
    """

    model = init_chat_model(
        name="respond",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )
    prompt = RESPONSE_SYSTEM_PROMPT.format(context=state.context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages, config)
    print(f"respond:{response}")

    return {"messages": [response]}
