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

logger = build_logger()


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

                  ):
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            unique_thread_id = f"thread_{uuid.uuid4()}"
            class KBChatState(TypedDict):
                messages: Annotated[list[BaseMessage], add_messages]
                context: str
                sources: str
                question: str
            async def retrieve_documents(state: KBChatState) -> KBChatState:
                last_message = state["messages"][-1].content
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
                    "question": last_message,
                }

            async def generate_response(state: KBChatState) -> KBChatState:
                # 这里只做检查和准备工作，不生成响应
                if not state["context"] or state["context"].strip() == "":
                    return {
                        "messages": [AIMessage(content="ERROR_NO_CONTEXT")],
                        "sources": state.get("sources", [])
                    }
                return state

            workflow = StateGraph(KBChatState)
            workflow.add_node("retrieve", retrieve_documents)
            workflow.add_node("generate", generate_response)
            workflow.add_edge(START, "retrieve")
            workflow.add_edge("retrieve", "generate")
            workflow.add_edge("generate", END)

            kb_app = workflow.compile()
            config = {"configurable": {"thread_id": unique_thread_id}}

            # 由于使用唯一线程ID，所以历史消息为空
            all_messages = [HumanMessage(content=query)]

            # 运行到 generate 节点完成
            final_state = await kb_app.ainvoke({"messages": all_messages}, config=config)

            # ===== 现在流式生成 LLM 响应 =====
            prompt_template = get_prompt_template("rag", prompt_name)
            all_messages = final_state["messages"]
            if len(all_messages) > 4:
                recent_messages = all_messages[-4:]
            else:
                recent_messages = all_messages

            history_messages = []
            for msg in recent_messages:
                if isinstance(msg, HumanMessage):
                    history_messages.append(History(role="user", content=msg.content).to_msg_template())
                elif isinstance(msg, AIMessage):
                    history_messages.append(History(role="assistant", content=msg.content).to_msg_template())

            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(history_messages + [input_msg])

            from utils import get_ChatOpenAI, get_config_models
            # 获取配置的模型信息
            model_info = get_config_models(model_name=model, model_type="llm")
            if model_info and model in model_info:
                logger.info(f"model_info:{model_info}")
                # 使用配置的平台模型
                model_config = model_info[model]
                llm = get_ChatOpenAI(
                    model_name=model,
                    temperature=0.7,
                    timeout=30,
                    openai_api_base=model_config["api_base_url"],
                    openai_api_key=model_config["api_key"],
                    max_tokens=1000,
                )
            else:
                # 如果配置中没有找到模型，尝试使用init_chat_model
                logger.warning(f"未找到模型 {model} 的配置，使用默认配置")
                llm = init_chat_model(
                    model,
                    temperature=0.7,
                    timeout=30
                )

            # ===== 使用 astream 逐 token 输出 =====
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