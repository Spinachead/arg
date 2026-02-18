from core.state_graph.states.main_graph.agent_state import AgentState
from core.prompts import RESPONSE_SYSTEM_PROMPT
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from settings import Settings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """
    根据上下文调用LLM大模型就行回复，支持工具调用
    """

    model = init_chat_model(
        name="respond",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history")
    ])
    chain = chat_prompt | model
    response = await chain.ainvoke({
        "history": state.messages,
        "context": state.context,
    }, config)
    print(f"respond:{response}")
    return {"messages": [response]}
