import json
import uuid
from typing import AsyncIterable, TypedDict, Annotated, List, Dict

from fastapi import Body, UploadFile, File, Form, Depends
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from openai import timeout
from sse_starlette import EventSourceResponse
from api_schemas import OpenAIChatOutput, ChatOptions, GraphChatRequest
from knowledge_base.kb_service.base import KBServiceFactory
from knowledge_base.model.kb_document_model import DocumentWithVSId
from utils import BaseResponse, build_logger, format_reference, get_prompt_template, History
from langchain_ollama import OllamaLLM, ChatOllama
import os
from dotenv import load_dotenv
from .rag import setup_rag_tools
from .auth_middleware import get_current_user
from utils import get_ChatOpenAI, get_config_models
from langsmith import traceable, get_current_run_tree

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
                  conversation_id: str = Body(None, description="对话ID，用于关联消息"),
                  options: ChatOptions = Body(None, description="配置选项"),
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
            
            # 用于收集完整的回答
            full_response = ""
            message_id = f"msg_{uuid.uuid4().hex}"
                        
            # 尝试获取 LangSmith trace_id
            trace_id = None
            try:
                from langsmith.run_helpers import get_current_run_tree
                run_tree = get_current_run_tree()
                if run_tree is not None:
                    trace_id = str(run_tree.trace_id)
            except Exception as e:
                logger.warning(f"无法获取 LangSmith trace_id: {e}")
            
            async for token in llm.astream(
                    chat_prompt.format(
                        context=final_state["context"],
                        sources=final_state["sources"] if final_state["sources"] else "未知来源",
                        question=final_state["question"],
                    )
            ):
                ret = OpenAIChatOutput(
                    id=message_id,
                    object="chat.completion.chunk",
                    content=token.content,  # 单个 token
                    role="assistant",
                    model=model,
                )
                ret_dict = ret.model_dump()
                ret_dict["sources"] = final_state.get("sources", [])
                            
                # 收集完整回答
                full_response += token.content
                            
                yield json.dumps(ret_dict, ensure_ascii=False)
                        
            # 流式输出结杞后，保存到数据库
            try:
                from db.repository.message_repository import add_message_to_db
                            
                # 如果没有传入 conversation_id，生成一个
                conv_id = (options.conversationId if options else None) or conversation_id or f"conv_{uuid.uuid4().hex}"
                            
                add_message_to_db(
                    message_id=message_id,
                    conversation_id=conv_id,
                    chat_type="kb_chat",
                    query=query,
                    response=full_response,
                    trace_id=trace_id,
                    meta_data={
                        "kb_name": kb_name,
                        "model": model,
                        "sources": final_state.get("sources", []),
                    },
                )
                logger.info(f"消息已保存到数据库: message_id={message_id}, trace_id={trace_id}")
            except Exception as e:
                logger.error(f"保存消息到数据库失败: {e}")
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


from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, AsyncIterable, Literal, List, TypedDict, Annotated


class ChatProcessRequest(BaseModel):
    prompt: str

async def chat_process(request: ChatProcessRequest):
    if not request.prompt:
        return {"status": "Error", "data": None, "message": "Question input is required"}

    try:
        prompt_template = get_prompt_template("rag", "default")
        history_message = History(role="user", content=prompt_template).to_msg_template(False)
        logger.info(f"history_message: {history_message}")
        chat_prompt = ChatPromptTemplate.from_messages([
                History(role="user", content=prompt_template).to_msg_template(False)
            ])
        logger.info(f"chat_prompt: {chat_prompt}")
        return {
            "status": "Error",
            "data": None,
            "message": "rag_chain 未初始化，请先调用 setup_rag_tools 函数"  
        }
    except Exception as e:
        return {"status": "Error", "data": None, "message": str(e)}

# 定义结构化输出模型（Pydantic）
class QueryWithKB(BaseModel):
    """查询变体及其对应的知识库"""
    query: str = Field(description="重写后的查询文本")
    kb_name: str = Field(description="最适合该查询的知识库名称")

class MultiQueryResult(BaseModel):
    """多个查询变体的结果"""
    queries: List[QueryWithKB] = Field(description="查询变体列表，包含查询文本和对应的知识库名称")

class RouteQuery(BaseModel):
    """路由用户查询到最合适的知识库名称。"""
    datasource: Literal["direct_answer", "company_docs", "law_faq"] = Field(
        description="选择路径: direct_answer(直接回答), company_docs(公司文档), product_faq(法律FAQ)"
    )

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
                  conversation_id: str = Body(None, description="对话ID，用于关联消息"),
                  options: ChatOptions = Body(None, description="配置选项")
                  ):
    @traceable(name="AgentChatIteration")
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            # 获取用户长期记忆和偏好
            user_profile = {}
            try:
                from db.repository.user_memory_repository import get_user_profile_from_memories
                # user_id = current_user.user_id if hasattr(current_user, 'user_id') else None
                # todo: 这里要通过current_user获取到userId
                user_id = 1
                if user_id:
                    user_profile = get_user_profile_from_memories(user_id=user_id)
                    logger.info(f"已加载用户 {user_id} 的长期记忆: {user_profile}")
            except Exception as e:
                logger.warning(f"获取用户记忆失败: {e}")
                user_profile = {"preferred_kbs": [], "preferred_domains": [], "tone": "标准", "recent_topics": []}
            
            class KBChatState(TypedDict):
                context: str
                sources: str
                question: str

            async def generate_queries(original_query: str) -> List[Dict[str, str]]:
                """生成多个查询变体及其对应的知识库名称"""
                from utils import get_ChatOpenAI, get_config_models
                from db.repository.knowledge_base_repository import list_kbs_from_db
                
                # 获取所有可用的知识库列表
                available_kbs = list_kbs_from_db()
                kb_info_str = "\n".join([f"- {kb.kb_name}: {kb.kb_info or '无描述'}" for kb in available_kbs])
                
                # 获取模型配置
                model_info = get_config_models(model_name=model, model_type="llm")
                model_config = model_info[model]
                llm = get_ChatOpenAI(
                    model_name=model,
                    openai_api_base=model_config["api_base_url"],
                    openai_api_key=model_config["api_key"],
                )
                
                # 使用结构化输出
                structured_llm = llm.with_structured_output(MultiQueryResult)
                
                # 生成查询变体和知识库匹配的提示
                query_gen_prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一个专业的查询分析和改写助手。根据用户的原始查询，你需要：
                    1. 生成3个不同角度或表述的查询变体，用于从知识库中检索相关信息
                    2. 为每个查询变体选择最合适的知识库

                    用户的长期偏好信息：
                    {user_profile}
                    
                    在选择知识库时，请优先考虑用户常用或偏好的知识库（preferred_kbs）和领域（preferred_domains）。

                    可用的知识库列表：{kb_list}

                    请返回JSON格式，包含queries数组，每个元素有query（查询文本）和kb_name（知识库名称）两个字段。"""),
                    ("human", "原始查询: {query}")
                ])
                
                # 调用模型生成查询变体和知识库匹配
                try:
                    result = await structured_llm.ainvoke(
                        query_gen_prompt.format(
                            query=original_query, 
                            kb_list=kb_info_str,
                            user_profile=json.dumps(user_profile, ensure_ascii=False)
                        )
                    )
                    
                    # 返回查询变体列表，每个元素包含query和kb_name
                    query_kb_pairs = [{"query": q.query, "kb_name": q.kb_name} for q in result.queries]
                    
                    # 如果没有生成足够的查询变体，添加原始查询
                    if len(query_kb_pairs) < 3:
                        query_kb_pairs.insert(0, {"query": original_query, "kb_name": kb_name})
                    
                    return query_kb_pairs
                except Exception as e:
                    logger.warning(f"结构化输出失败，使用备用方案: {e}")
                    # 备用方案：返回原始查询
                    return [{"query": original_query, "kb_name": kb_name}]
            
            async def multi_query_retrieve(state: KBChatState) -> KBChatState:
                """使用多个查询变体检索文档并合并结果"""
                # 生成多个查询变体
                query_kb_pairs = await generate_queries(query)
                
                all_docs = []
                doc_id_set = set()
                
                # 对每个查询变体执行搜索
                for pair in query_kb_pairs:
                    q = pair["query"]
                    target_kb = pair["kb_name"]
                    
                    # 如果知识库名称有效，使用指定的知识库，否则使用默认知识库
                    kb_to_use = target_kb if target_kb else kb_name
                    
                    # logger.info(f"查询: '{q}' -> 知识库: {kb_to_use}")
                    
                    docs = search_docs(
                        query=q,
                        knowledge_base_name=kb_to_use,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        file_name="",
                        metadata={}
                    )
                    
                    # 合并结果，去重
                    for doc in docs:
                        doc_id = doc.get("id") or doc.get("metadata", {}).get("id")
                        if doc_id and doc_id not in doc_id_set:
                            doc_id_set.add(doc_id)
                            all_docs.append(doc)
                
                # 如果没有找到文档，使用原始查询再试一次
                if not all_docs:
                    docs = search_docs(
                        query=query,
                        knowledge_base_name=kb_name,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        file_name="",
                        metadata={}
                    )
                    all_docs = docs
                
                # 格式化结果
                source_documents = format_reference(kb_name, all_docs, "")
                context = "\n\n".join([doc.get("page_content", "") for doc in all_docs])
                
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
            workflow.add_node("retrieve", multi_query_retrieve)
            workflow.add_node("generate", generate_response)
            workflow.add_edge(START, "retrieve")
            workflow.add_edge("retrieve", "generate")
            workflow.add_edge("generate", END)
            kb_app = workflow.compile()
            final_state = await kb_app.ainvoke({"context": "", "sources": "", "question": query})
            
            # prompt_template = get_prompt_template("rag", prompt_name)
            from langsmith import Client
            client = Client()
            chat_prompt = client.pull_prompt("arg_default:9e93ae37")
            # logger.info(f"Using prompt template: {prompt_template}")
            # chat_prompt = ChatPromptTemplate.from_messages([
            #     History(role="user", content=prompt_template).to_msg_template(False)
            # ])
            
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
            
            # 用于收集完整的回答
            full_response = ""
            message_id = f"msg_{uuid.uuid4().hex}"
            conv_id = (options.conversationId if options else None) or conversation_id or f"conv_{uuid.uuid4().hex}"
            trace_id = None
            run_tree = get_current_run_tree()
            if run_tree is not None:
                trace_id = str(run_tree.trace_id)
            
            messages = chat_prompt.format_messages(
                context=final_state["context"],
                sources=final_state["sources"] if final_state["sources"] else "未知来源",
                question=final_state["question"],
                user_profile=json.dumps(user_profile, ensure_ascii=False),
            )

            async for token in llm.astream(messages):
                ret = OpenAIChatOutput(
                    id=message_id,
                    object="chat.completion.chunk",
                    content=token.content,  # 单个 token
                    role="assistant",
                    model=model,
                    message_id=message_id,
                )
                ret_dict = ret.model_dump()
                ret_dict["sources"] = final_state.get("sources", [])
                ret_dict["conversationId"] = conv_id

                            
                # 收集完整回答
                full_response += token.content
                            
                yield json.dumps(ret_dict, ensure_ascii=False)
                        
            # 流式输出结杞后，保存到数据库
            try:
                from db.repository.message_repository import add_message_to_db
                            
                add_message_to_db(
                    message_id=message_id,
                    conversation_id=conv_id,
                    chat_type="agent_kb_chat",
                    query=query,
                    response=full_response,
                    trace_id=trace_id,
                    meta_data={
                        "kb_name": kb_name,
                        "model": model,
                        "sources": final_state.get("sources", []),
                        "user_profile": user_profile,  # 保存用户画像
                    },
                )
                logger.info(f"消息已保存到数据库: message_id={message_id}, trace_id={trace_id}")
            except Exception as e:
                logger.error(f"保存消息到数据库失败: {e}")
        except Exception as e:
            logger.exception(e)
            yield json.dumps({"error": str(e)})
            return
    return EventSourceResponse(knowledge_base_chat_iterator())


async def graph_chat(request: GraphChatRequest = Body(..., description="Graph Chat 请求体")):
    @traceable(name="GraphChatIteration")
    async def graph_chat_iterator() -> AsyncIterable[str]:
        try:
            from chat.graph_chat import create_graph_chat_app
            # 创建graph应用
            graph_app = create_graph_chat_app()
            
            # 运行graph获取检索结果（user_profile将由LLM通过tool自主获取）
            initial_state = {
                "query": request.query,
                "kb_name": request.kb_name,
                "top_k": request.top_k,
                "score_threshold": request.score_threshold,
                "model": request.model,
                "temperature": request.temperature,
                "user_id": 1,  # TODO: 从current_user获取
                "user_profile": {},  # 空字典，将由LLM决定是否通过tool加载
                "query_kb_pairs": [],
                "context": "",
                "sources": [],
            }
            messages = graph_app.invoke(initial_state, {"configurable": {"thread_id": "1"}})
            for message in messages:
                message.pretty_print()
          
        except Exception as e:
            logger.exception(e)
            yield json.dumps({"error": str(e)})
            return
    
    return EventSourceResponse(graph_chat_iterator())
                



# 对回复打分
async def score_reply(conversation_id: str = Body(None, description="对话ID，用于关联消息"),
                    score: int = Body(1, description="评分，范围0-100"),
                    message_id: str = Body(None, description="消息ID，用于关联消息"),
                    reason: str = Body("", description="评分理由"))->BaseResponse:
    """
    对指定对话的最新回复进行打分
    :param conversation_id: 对话ID
    :param score: 评分，0-100之间，越高表示越满意
    :param reason: 评分理由（可选）
    :return: 操作结果
    """
    try:
        from db.repository.message_repository import update_message_feedback, get_message_by_id
        
        if not conversation_id:
            return {"code": 400, "msg": "对话ID不能为空"}
        
        # 验证评分范围
        if score < 0 or score > 100:
            return {"code": 400, "msg": "评分必须在0-100之间"}
        
        # 获取该对话的最新消息
        message = get_message_by_id(message_id=message_id)
        
        if not message:
            return {"code": 404, "msg": "未找到该对话的消息记录"}
        
        
        # 更新消息评分
        success = update_message_feedback(
            message_id=message_id,
            feedback_score=score,
            feedback_reason=reason
        )
        
        if success:
            logger.info(f"消息评分更新成功: conversation_id={conversation_id}, message_id={message_id}, score={score}")
            return BaseResponse(code=200, msg="评分成功", data={"message_id": message_id, "conversation_id": conversation_id, "score": score, "reason": reason})    
        else:
            return BaseResponse(code=500, msg="评分失败，消息不存在")
    except Exception as e:
        logger.exception(e)
        return BaseResponse(code=500, msg=f"评分失败: {str(e)}")



async def auto_route(query: str = Body(..., description="用户输入", example=["你好"])):
    try:
        # 获取模型配置
        model_info = get_config_models(model_name=model, model_type="llm")
        model_config = model_info[model]
        llm = get_ChatOpenAI(
                model_name=model,
                temperature=0.7,
                timeout=30,
                openai_api_base=model_config["api_base_url"],
                openai_api_key=model_config["api_key"],
                max_tokens=500,
            )
        # 初始化LLM（用structured output）
        structured_llm = llm.with_structured_output(RouteQuery)
        classify_prompt = ChatPromptTemplate.from_template(
            """分析以下查询，选择最合适的路径：
            - 如果是简单常识或不需要外部知识，直接用'direct_answer'。
            - 如果涉及公司内部文档，用'company_docs'。
            - 如果是产品FAQ，用'product_faq'。
            查询: {query}
            输出JSON格式。"""
        )
        # 分类链
        classifier_chain = classify_prompt | structured_llm | JsonOutputParser()
        # 定义不同路径的链（假设你已有RAG链）
        from langchain_core.runnables import RunnableBranch, RunnablePassthrough
        direct_answer_chain = ChatPromptTemplate.from_template("直接回答: {query}") | llm
        company_rag_chain = RunnablePassthrough.assign(docs=company_retriever) | rag_prompt | llm  # 你的RAG链
        product_rag_chain = RunnablePassthrough.assign(docs=product_retriever) | rag_prompt | llm
        # 路由分支
        router = RunnableBranch(
            (lambda x: x["datasource"] == "direct_answer", direct_answer_chain),
            (lambda x: x["datasource"] == "company_docs", company_rag_chain),
            (lambda x: x["datasource"] == "product_faq", product_rag_chain),
            direct_answer_chain  # 默认路径
        )
        # 完整链：分类 + 路由
        full_chain = (
            {"query": RunnablePassthrough(), "datasource": classifier_chain}
            | router
        )
        # 使用
        result = full_chain.invoke({"query": "公司假期政策是什么？"})  # 会路由到company_docs
    except Exception as e:
        logger.exception(e)
        return {"error": str(e)}