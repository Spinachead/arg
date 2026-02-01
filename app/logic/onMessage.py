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
        kb_name = pair.get("kb_name") or pair.get("type")
        await current_step.stream_token(
            f"{i+1}. **{kb_name}**: {query}\n"
        )


@cl.step(type="retrieval", name="知识检索", show_input=False)
async def retrieve_documents_step(data: dict):
    sources = data.get("sources", [])
    count = len(sources) if isinstance(sources, list) else 0
    cl.context.current_step.output = f"检索完成：共找到 {count} 份相关内容"
   

# @traceable(name="on_message")
async def execute1(message: cl.Message):
    graph: Runnable = cl.user_session.get("graph")
    state: InputState = cl.user_session.get("state")
    question = message.content
    state.messages += [HumanMessage(content=question)]
    
    ui_message = cl.Message(content="")
    
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

async def execute(message: cl.Message):
    actions = [
        cl.Action(
            name="action_button",
            icon="mouse-pointer-click",
            payload={"value": "example_value"},
            label="Click me!"
        )
    ]
    ui_message = cl.Message(content="", actions=actions)
    await ui_message.send()
    ui_message.content = "Hello, how can I help you?"
    await ui_message.update()
