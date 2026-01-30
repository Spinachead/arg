import chainlit as cl
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from core.state_graph.states.main_graph.router import Router
from core.state_graph.states.main_graph.input_state import InputState
from langsmith import traceable, get_current_run_tree
from db.repository.message_repository import add_message_to_db

@cl.step(type="llm", name="查询变体", show_input=False)
async def generate_queries_step(plan: dict):
    current_step = cl.context.current_step
    steps = plan.get("steps") or plan.get("query_kb_pairs") or []
    for i, step in enumerate(steps):
        question = step.get("question") or step.get("query")
        stype = step.get("type") or step.get("kb_name")
        await current_step.stream_token(
            f"{i+1}. **{stype}**: {question}\n"
        )


@cl.step(type="llm", name="Classify Request", show_input=False)
async def classification_step(classification: Router):
    current_step = cl.context.current_step
    await current_step.stream_token(
        f"Classified as **{classification.type}** with the logic: _{classification.logic}_"
    )

@traceable(name="on_message")
async def execute(message: cl.Message):
    graph: Runnable = cl.user_session.get("graph")
    state: InputState = cl.user_session.get("state")
    question = message.content
    state.messages += [HumanMessage(content=question)]
    ui_message = cl.Message(content="")
    await ui_message.send()
    async for event in graph.astream_events(state, version="v2"):
        event_name = event["name"]
        event_event = event["event"]
        if event_event == "on_chain_end":
            if event_name == "generate_queries":
                steps = event.get("data").get("output")
                print(f"这是steps:{steps}")
                await generate_queries_step(steps)
            
        if event["event"] == "on_chain_end" and event["name"] == "test":
          
            if event.get("data") is not None and isinstance(event["data"], dict):
                # First try to get from output (ChatGPT compatibility)
                output = event["data"].get("output")
                if output is not None and isinstance(output, dict):
                    router = output.get("router")
                    if router is not None:
                        await classification_step(router)
                        continue
                
                input_data = event["data"].get("input")
                if input_data is not None and hasattr(input_data, "router"):
                    router = input_data.router
                    if router is not None:
                        await classification_step(router)

        if event["name"] == "analyze_and_route_query": 
            classification = event["data"]["output"]["router"]
            await classification_step(classification)
        
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                await ui_message.stream_token(content)

        if event["event"] == "on_chain_end":
            output = event['data'].get('output')
            if isinstance(output, dict) and "messages" in output:
                last_message = output["messages"][-1]
                if isinstance(last_message, AIMessage):
                    ui_message.content = last_message.content
    await ui_message.update()

    state.messages += [AIMessage(content=ui_message.content)]

    # 将消息存入数据库以实现历史记录持久化
    try:
        user = cl.user_session.get("user")
        conversation_id = user.identifier if user else cl.user_session.get("id")
        add_message_to_db(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,

            chat_type="graph_chat",  # 或者根据实际情况设置
            query=question,
            response=ui_message.content,
            meta_data={}
        )
    except Exception as e:
        print(f"Error saving message to DB: {e}")
