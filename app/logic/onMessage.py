import chainlit as cl
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from core.state_graph.states.main_graph.router import Router
from core.state_graph.states.main_graph.input_state import InputState
from langsmith import traceable, get_current_run_tree
from db.repository.message_repository import add_message_to_db

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
    state: InputState = cl.user_session.get("state")
    
    # # 安全检查：如果 state 或 graph 为 None（通常发生在服务器重启后的会话恢复），则重新初始化
    # if state is None or graph is None:
    #     from core.main_graph import build_main_graph
    #     from core.state_graph.states.main_graph.input_state import InputState
        
    #     graph = build_main_graph()
    #     state = InputState(messages=[])
        
    #     cl.user_session.set("graph", graph)
    #     cl.user_session.set("state", state)
    question = message.content
    state.messages += [HumanMessage(content=question)]
    upload_action = cl.Action(
        name="upload_document",     
        label="上传文档",
        value="any_value",
        icon="upload",
        payload={"value": "example_value"}
    )
    
    ui_message = cl.Message(content="", actions=[upload_action])
    
    async for event in graph.astream_events(state, version="v2"):
        
        if event["event"] == "on_chain_end":
            if event["name"] == "generate_queries":
                await generate_queries_step(event.get("data").get("output"))
            if event["name"] == "retrieve_documents":
                await retrieve_documents_step(event.get("data").get("output"))

        if event["event"] == "on_chat_model_stream":
            if not ui_message.id:
                await ui_message.send()
            
            content = event["data"]["chunk"].content
            if content:
                await ui_message.stream_token(content)
    
    await ui_message.update()
    # 最后同步状态
    state.messages += [AIMessage(content=ui_message.content)]

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
        response=ui_message.content,
        message_id=ui_message.id,
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
