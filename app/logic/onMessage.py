import chainlit as cl
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from core.state_graph.states.main_graph.router import Router
from core.state_graph.states.main_graph.input_state import InputState
from langsmith import traceable, get_current_run_tree
from db.repository.message_repository import add_message_to_db
from langchain_core.runnables.config import RunnableConfig
from settings import Settings


@cl.step(type="llm", name="查询优化", show_input=False)
async def generate_queries_step(data: dict):
    current_step = cl.context.current_step
    pairs = data.get("query_kb_pairs") or data.get("steps") or []
    for i, pair in enumerate(pairs):
        query = pair.get("query") or pair.get("question")
        await current_step.stream_token(
            f"{i+1}. {query}\n"
        )


@cl.step(type="retrieval", name="知识检索", show_input=False)
async def retrieve_documents_step(data: dict):
    sources = data.get("sources", [])
    count = len(sources) if isinstance(sources, list) else 0
    cl.context.current_step.output = f"检索完成：共找到 {count} 份相关内容"
   

# @traceable(name="on_message")
async def execute(message: cl.Message):
    graph: Runnable = cl.user_session.get("graph")
    user = cl.user_session.get("user")
    model_settings = cl.user_session.get("model_settings", {})
    kb_name = model_settings.get("knowledge_base", Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE)
    
    question = message.content
    
    input_state = InputState(messages=[HumanMessage(content=question)])
    
    upload_action = cl.Action(
        name="upload_document",     
        label="上传文档",
        value="any_value",
        icon="upload",
        payload={"kb_name": kb_name}
    )
    
    # 用于追踪当前正在流式输出的消息气泡
    active_ui_message = None
    last_run_id = None
    
    # 使用 thread_id 来关联对话历史，checkpoint 会自动加载/保存
    config = RunnableConfig({"configurable": {"thread_id": user.id}})

    async for event in graph.astream_events(input_state, version="v2", config=config):
        print(f"event:{event}")
        
        if event["event"] == "on_chain_end":
            if event["name"] == "generate_queries":
                await generate_queries_step(event.get("data").get("output"))
            if event["name"] == "retrieve_documents":
                await retrieve_documents_step(event.get("data").get("output"))

        if event["event"] == "on_chat_model_stream":
            run_id = event["run_id"]
            
            # 如果 run_id 变化，说明是新的一轮模型调用（例如工具执行后的第二次回复）
            if run_id != last_run_id:
                if active_ui_message:
                    await active_ui_message.update()
                # 创建新的消息气泡，确保它出现在工具执行 Step 的下方
                active_ui_message = cl.Message(content="")
                last_run_id = run_id

            content = event["data"]["chunk"].content
            if content:
                if not active_ui_message.id:
                    await active_ui_message.send()
                await active_ui_message.stream_token(content)
    
    # 确保最后有一个消息，并带上操作按钮
    if active_ui_message:
        active_ui_message.actions = [upload_action]
        await active_ui_message.update()
    else:
        active_ui_message = cl.Message(content="已完成处理", actions=[upload_action])
        await active_ui_message.send()

    # 获取 LangSmith Trace ID (如果有)
    trace_id = None
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            trace_id = str(run_tree.id)
    except Exception:
        pass

    # 保存对话到数据库
    user = cl.user_session.get("user")
    user_id = getattr(user, "id", None)
    add_message_to_db(
        thread_id=message.thread_id,
        query=question,
        response=active_ui_message.content,
        message_id=active_ui_message.id,
        trace_id=trace_id,
        chat_type="graph_chat",  # 或者根据实际逻辑调整
        metadata={"user_id": user_id}
    )


async def execute1(message: cl.Message):
    user = cl.user_session.get("user")
    if user:
        user_id = getattr(user, "id", None)
        print(f"user_id: {user_id}")
    else:
        print("user is None")

    upload_action = cl.Action(
        name="upload_document",     
        label="上传文档",
        value="any_value",
        icon="upload",
        payload={"value": "example_value"}
    )
    ui_message = cl.Message(content="", actions=[upload_action])
    await ui_message.send()
    ui_message.content = "Hello, how can I help you?"
    await ui_message.update()
