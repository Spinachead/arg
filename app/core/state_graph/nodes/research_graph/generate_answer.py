from core.state_graph.states.research_graph.query_state import QueryState
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from settings import Settings
from langchain_core.messages import AIMessage

def generate_answer(state: QueryState) -> dict:
    """
    把sql查询结果生成自然语言回答
    """
    model = init_chat_model(
        name="generate_answer",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )

    if state.get("sql_result") is None:
        # 直接回答
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    
    # 基于查询结果生成自然语言回答
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手，请根据查询结果用中文简洁回答用户问题。"),
        ("human", "用户问题：{question}\n查询结果：{result}"),
    ])
    chain = prompt | model
    question = state["messages"][-1].content
    result_str = str(state["query_result"])[:1000]  # 防止过长
    response = chain.invoke({"question": question, "result": result_str})
    return {"messages": state["messages"] + [AIMessage(content=response.content)]}