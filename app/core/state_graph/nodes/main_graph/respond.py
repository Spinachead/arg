from core.state_graph.states.main_graph.agent_state import AgentState
from core.prompts import RESPONSE_SYSTEM_PROMPT
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from settings import Settings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def filter_messages_for_response(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    过滤消息列表，只保留用户问题和最终的 AI 回答，去除工具调用相关的消息。
    这样可以避免模型重复生成工具调用格式。
    """
    filtered = []
    for msg in messages:
        # 保留用户消息
        if isinstance(msg, HumanMessage):
            filtered.append(msg)
        # 保留没有 tool_calls 的 AI 消息（即最终回答）
        elif isinstance(msg, AIMessage):
            if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                # 只保留有实际内容的 AI 消息
                if msg.content and msg.content.strip():
                    filtered.append(msg)
    return filtered


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """
    根据上下文调用LLM大模型进行回复
    """

    model = init_chat_model(
        name="respond",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )

    # 过滤掉工具调用相关的消息，避免模型重复生成工具调用格式
    filtered_messages = filter_messages_for_response(state.messages)

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history")
    ])
    chain = chat_prompt | model
    response = await chain.ainvoke({
        "history": filtered_messages,
        "context": state.context,
    }, config)
    print(f"respond:{response}")
    return {"messages": [response]}
