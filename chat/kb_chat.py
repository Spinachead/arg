import json
import uuid
from typing import AsyncIterable, TypedDict, Annotated, List, Dict

from fastapi import Body, UploadFile, File, Form
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from openai import timeout
from sse_starlette import EventSourceResponse

from api_schemas import OpenAIChatOutput
from knowledge_base.kb_service.base import KBServiceFactory
from knowledge_base.model.kb_document_model import DocumentWithVSId
from utils import build_logger, format_reference, get_prompt_template, History
from langchain_ollama import OllamaLLM, ChatOllama
import os
from dotenv import load_dotenv
from .rag import setup_rag_tools
load_dotenv()

logger = build_logger()
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")  # 从 LangSmith 复制
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

async def kb_chat(query: str = Body(..., description="用户输入", example=["你好"]),
                  top_k: int = Body(3, description="匹配向量数字"),
                  score_threshold: float = Body(
                      0.5,
                      description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                      ge=0,
                      le=2,
                  ),
                  kb_name: str = Body("",
                                      description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称",
                                      examples=["samples"]),
                  prompt_name: str = Body(
                      "default",
                      description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                  ),
                  model: str = Body("qwen-max", description="LLM 模型名称。"),
                  temperature: float = Body(0.5, description="温度"),
                  ):
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            class KBChatState(TypedDict):
                context: str
                sources: str
                question: str
            
            async def retrieve_documents(state: KBChatState) -> KBChatState:
                docs = search_docs(
                    query=query,
                    knowledge_base_name=kb_name,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    file_name="",
                    metadata={}
                )
                source_documents = format_reference(kb_name, docs, "")
                context = "\n\n".join([doc.get("page_content", "") for doc in docs])

                return {
                    "context": context,
                    "sources": source_documents,
                    "question": query,
                }

            async def generate_response(state: KBChatState) -> KBChatState:
                if not state["context"] or state["context"].strip() == "":
                    return {
                        "context": "",
                        "sources": state.get("sources", []),
                        "question": state.get("question", ""),
                    }
                return state

            workflow = StateGraph(KBChatState)
            workflow.add_node("retrieve", retrieve_documents)
            workflow.add_node("generate", generate_response)
            workflow.add_edge(START, "retrieve")
            workflow.add_edge("retrieve", "generate")
            workflow.add_edge("generate", END)
            kb_app = workflow.compile()
            final_state = await kb_app.ainvoke({"context": "", "sources": "", "question": query})

            prompt_template = get_prompt_template("rag", prompt_name)
            chat_prompt = ChatPromptTemplate.from_messages([
                History(role="user", content=prompt_template).to_msg_template(False)
            ])

            from utils import get_ChatOpenAI, get_config_models
            model_info = get_config_models(model_name=model, model_type="llm")
            model_config = model_info[model]
            llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                timeout=30,
                openai_api_base=model_config["api_base_url"],
                openai_api_key=model_config["api_key"],
                max_tokens=1000,
            )
            

            async for token in llm.astream(
                    chat_prompt.format(
                        context=final_state["context"],
                        sources=final_state["sources"] if final_state["sources"] else "未知来源",
                        question=final_state["question"],
                    )
            ):
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion.chunk",
                    content=token.content,  # 单个 token
                    role="assistant",
                    model=model,
                )
                ret_dict = ret.model_dump()
                ret_dict["sources"] = final_state.get("sources", [])
                yield json.dumps(ret_dict, ensure_ascii=False)
        except Exception as e:
            logger.exception(e)
            yield json.dumps({"error": str(e)})
            return
    return EventSourceResponse(knowledge_base_chat_iterator())



def search_docs(
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(
            ..., description="知识库名称", examples=["samples"]
        ),
        top_k: int = Body(3, description="匹配向量数"),
        score_threshold: float = Body(
            2.0,
            description="知识库匹配相关度阈值，取值范围在0-1之间，"
                        "SCORE越小，相关度越高，"
                        "取到2相当于不筛选，建议设置在0.5左右",
            ge=0.0,
            le=2.0,
        ),
        file_name: str = Body("", description="文件名称，支持 sql 通配符"),
        metadata: dict = Body({}, description="根据 metadata 进行过滤，仅支持一级键"),
) -> List[Dict]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    data = []
    if kb is not None:
        if query:
            docs = kb.search_docs(query, top_k, score_threshold)
            # data = [DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")) for x in docs]
            data = [DocumentWithVSId(**{"id": x.metadata.get("id"), **x.dict()}) for x in docs]
            logger.info(f"search_docs:{docs}")
        elif file_name or metadata:
            data = kb.list_docs(file_name=file_name, metadata=metadata)
            for d in data:
                if "vector" in d.metadata:
                    del d.metadata["vector"]
            logger.info(f"search_docs_data:{data}")
    return [x.dict() for x in data]


from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncIterable, Literal, List, TypedDict, Annotated
from rag_chain import create_rag_graph
rag_chain = create_rag_graph()


class ChatProcessRequest(BaseModel):
    prompt: str
    options: Optional[Dict[str, Any]] = {}
    systemMessage: Optional[str] = ""
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

async def chat_process(request: ChatProcessRequest):
    if not request.prompt:
        return {"status": "Error", "data": None, "message": "Question input is required"}

    try:
        # 获取会话ID，用于记忆功能
        session_id = request.options.get("sessionId", "default_session") if request.options else "default_session"
        config = {"configurable": {"thread_id": session_id}}

        # 使用带记忆功能的 RAG 链处理请求
        input_data = {
            "messages": [HumanMessage(content=request.prompt)]
        }
        response = rag_chain.invoke(input_data, config=config)

        return {
            "status": "Success",
            "data": {
                "id": "chat-1",
                "role": "assistant",
                "text": response["messages"][-1].content,
                "dateTime": "1111111"
            },
            "message": "Success"
        }
    except Exception as e:
        return {"status": "Error", "data": None, "message": str(e)}

async def agent_chat(query: str = Body(..., description="用户输入", example=["你好"]),
                  top_k: int = Body(3, description="匹配向量数字"),
                  score_threshold: float = Body(
                      0.5,
                      description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                      ge=0,
                      le=2,
                  ),
                  kb_name: str = Body("",
                                      description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称",
                                      examples=["samples"]),
                  prompt_name: str = Body(
                      "default",
                      description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                  ),
                  model: str = Body("qwen-max", description="LLM 模型名称。"),
                  temperature: float = Body(0.5, description="温度"),
                  ):
        async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
            try:
                from utils import get_ChatOpenAI, get_config_models
                from langgraph.prebuilt import create_react_agent
                from langgraph.checkpoint.memory import InMemorySaver
                from langchain_core.messages import HumanMessage

                model_info = get_config_models(model_name=model, model_type="llm")
                model_config = model_info[model]
                llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                timeout=30,
                openai_api_base=model_config["api_base_url"],
                openai_api_key=model_config["api_key"],
                max_tokens=1000,
                )
                rag_tools = await setup_rag_tools()

                # 使用 LangGraph 创建异步代理
                checkpointer = InMemorySaver()
                agent = create_react_agent(
                    model=llm,
                    tools=rag_tools,
                    checkpointer=checkpointer,
                )

                # 创建配置
                config = {"configurable": {"thread_id": f"agent_chat_{uuid.uuid4()}"}}

                # 使用异步流式处理
                async for event in agent.astream(
                    {"messages": [HumanMessage(content=query)]},
                    config=config,
                    stream_mode="messages"
                ):
                    # 检查是否是 AI 消息
                    if hasattr(event, 'content') and event.content:
                        ret = OpenAIChatOutput(
                            id=f"chat{uuid.uuid4()}",
                            object="chat.completion.chunk",
                            content=event.content,
                            role="assistant",
                            model=model,
                        )
                        ret_dict = ret.model_dump()
                        yield json.dumps(ret_dict, ensure_ascii=False)

            except Exception as e:
                logger.exception(e)
                yield json.dumps({"error": str(e)})
                return
        return EventSourceResponse(knowledge_base_chat_iterator())