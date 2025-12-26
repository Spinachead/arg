import json
import uuid
from typing import AsyncIterable, TypedDict, Annotated, List, Dict

from fastapi import Body, UploadFile, File, Form
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from sse_starlette import EventSourceResponse

from api_schemas import OpenAIChatOutput
from knowledge_base.kb_service.base import KBServiceFactory
from knowledge_base.model.kb_document_model import DocumentWithVSId
from utils import build_logger, format_reference, get_prompt_template, History
from langchain_ollama import OllamaLLM, ChatOllama

logger = build_logger()


async def kb_chat2(query: str = Body(..., description="用户输入", example=["你好"]),
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
                  model: str = Body("qwen:1.8b", description="LLM 模型名称。"),

                  ):
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            async with aiosqlite.connect("checkpoints.sqlite") as conn:
                checkpointer = AsyncSqliteSaver(conn)
                # checkpointer.setup()

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
                    logger.info(f"这是docs{docs}")
                    source_documents = format_reference(kb_name, docs, "")
                    context = "\n\n".join([doc.get("page_content", "") for doc in docs])

                    return {
                        "context": context,
                        "sources": source_documents,
                        "question": last_message,
                    }

                async def generate_response(state: KBChatState) -> KBChatState:
                    if not state["context"] or state["context"].strip() == "":
                        response = "根据提供的资料无法回答您的问题。知识库中不包含相关信息。"
                        return {
                            "messages": [AIMessage(content=response)],
                            "sources": state.get("sources", [])
                        }

                    prompt_template = get_prompt_template("rag", prompt_name)

                    # 限制历史消息数量，只保留最近的4条消息（2轮对话）
                    all_messages = state["messages"]
                    if len(all_messages) > 4:
                        recent_messages = all_messages[-4:]
                        logger.info(f"历史消息过多，仅保留最近4条消息")
                    else:
                        recent_messages = all_messages

                    # 构建历史消息列表，正确处理各种消息类型
                    history_messages = []
                    for msg in recent_messages:
                        if isinstance(msg, HumanMessage):
                            history_messages.append(History(role="user", content=msg.content).to_msg_template())
                        elif isinstance(msg, AIMessage):
                            history_messages.append(History(role="assistant", content=msg.content).to_msg_template())

                    # 添加当前问题的模板
                    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
                    chat_prompt = ChatPromptTemplate.from_messages(history_messages + [input_msg])
                    llm = ChatOllama(
                        model="qwen:1.8b",
                        temperature=0.7,
                    )
                    chain = chat_prompt | llm | StrOutputParser()

                    try:
                        
                        response = await chain.ainvoke({
                            "context": state["context"],
                            "sources": state["sources"] if state["sources"] else "未知来源",
                            "question": state["question"],
                        })
                        if not response:
                            response = "无法生成答案，请稍后重试。"
                        # 保留 sources 字段在 state 中，以便在响应中返回
                        return {
                            "messages": [AIMessage(content=response)],
                            "sources": state.get("sources", [])
                        }

                    except Exception as e:
                        logger.error(f"LLM调用失败: {str(e)}")
                        return {
                            "messages": [AIMessage(content=f"处理过程中出错: {str(e)}")],
                            "sources": state.get("sources", [])
                        }

                workflow = StateGraph(KBChatState)
                workflow.add_node("retrieve", retrieve_documents)
                workflow.add_node("generate", generate_response)
                workflow.add_edge(START, "retrieve")
                workflow.add_edge("retrieve", "generate")
                workflow.add_edge("generate", END)

                kb_app = workflow.compile(checkpointer=checkpointer)

                config = {"configurable": {"thread_id": "default_thread"}}

                # 关键：从数据库读取历史消息
                state_snapshot = await checkpointer.aget(config)
                # logger.info(f"stage_snapshot:{state_snapshot.get('channel_values', {})}")
                history_messages = state_snapshot.get('channel_values', {}).get('messages',
                                                                                []) if state_snapshot else []
                # 新消息追加到历史后面
                all_messages = history_messages + [HumanMessage(content=query)]

                inputs = {"messages": all_messages}

                sources_info = None
                async for event in kb_app.astream(inputs, stream_mode="values", config=config):
                    #todo:这里可以简化一下 参考https://docs.langchain.com/oss/python/langchain/agents  streaming
                    if isinstance(event, dict):
                        # 保存文档来源信息（从 state 中获取）
                        if "sources" in event:
                            sources_info = event.get("sources", [])
                        
                        # 处理消息响应
                        if "messages" in event:
                            messages = event["messages"]
                            if messages:
                                latest_message = messages[-1]
                                if isinstance(latest_message, AIMessage):
                                    content = latest_message.content
                                    logger.info(f"最终输出: {content[:100]}")

                                    if not isinstance(content, str):
                                        content = str(content)

                                    ret = OpenAIChatOutput(
                                        id=f"chat{uuid.uuid4()}",
                                        object="chat.completion.chunk",
                                        content=content,
                                        role="assistant",
                                        model=model,
                                    )
                                    # 添加文档来源信息到响应中
                                    ret_dict = ret.model_dump()
                                    # 从当前 event 或之前保存的 sources_info 中获取文档来源
                                    current_sources = event.get("sources", sources_info)
                                    if current_sources:
                                        ret_dict["sources"] = current_sources
                                        sources_info = current_sources  # 更新保存的 sources_info
                                    yield json.dumps(ret_dict, ensure_ascii=False)

        except Exception as e:
            logger.exception(e)
            yield json.dumps({"error": str(e)})
            return
    return EventSourceResponse(knowledge_base_chat_iterator())



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
                  model: str = Body("qwen:1.8b", description="LLM 模型名称。"),

                  ):
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            async with aiosqlite.connect("checkpoints.sqlite") as conn:
                checkpointer = AsyncSqliteSaver(conn)

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

            # ===== 改动：生成不返回完整响应，只返回检索结果 =====
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

            kb_app = workflow.compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "default_thread"}}

            state_snapshot = await checkpointer.aget(config)
            history_messages = state_snapshot.get('channel_values', {}).get('messages', []) if state_snapshot else []
            all_messages = history_messages + [HumanMessage(content=query)]

            # ===== 关键：先运行检索，然后逐 token 流式生成 =====
            # 运行到 generate 节点完成
            final_state = await kb_app.ainvoke({"messages": all_messages}, config=config)

            if "ERROR_NO_CONTEXT" in final_state["messages"][-1].content:
                yield json.dumps({
                    "id": f"chat{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "content": "根据提供的资料无法回答您的问题。",
                    "role": "assistant",
                    "model": model,
                    "sources": final_state.get("sources", [])
                })
                return

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
            
            llm = ChatOllama(
                model="qwen:1.8b",
                temperature=0.7,
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
                    content=token,  # 单个 token
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
        elif file_name or metadata:
            data = kb.list_docs(file_name=file_name, metadata=metadata)
            for d in data:
                if "vector" in d.metadata:
                    del d.metadata["vector"]
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