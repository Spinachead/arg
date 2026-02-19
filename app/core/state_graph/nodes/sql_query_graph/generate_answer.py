from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from settings import Settings
from langchain_core.messages import AIMessage
from core.state_graph.states.sql_query_graph.sql_query import SQLQueryState
from langchain.chat_models import init_chat_model
from core.prompts import SQL_GENERATE_ANSWER_PROMPT


def generate_answer(state: SQLQueryState, *, config: RunnableConfig) -> dict:
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
    
    # 基于查询结果生成自然语言回答
    prompt = ChatPromptTemplate.from_messages([
        ("system", SQL_GENERATE_ANSWER_PROMPT),
        MessagesPlaceholder("history"),
    ])
    chain = prompt | model
    response = chain.invoke({
        "history": state.messages,
    }, config)
    print(f"generate_answer: {response}")
    return {"context": response.content}
   
