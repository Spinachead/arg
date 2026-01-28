import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from core.state_graph.states.main_graph.router import Router
from core.state_graph.states.main_graph.input_state import InputState


@cl.step(type="llm", name="Classify Request", show_input=False)
async def classification_step(classification: Router):
    current_step = cl.context.current_step
    await current_step.stream_token(
        f"Classified as **{classification.type}** with the logic: _{classification.logic}_"
    )


async def execute(message: cl.Message):
    graph: Runnable = cl.user_session.get("graph")
    state: InputState = cl.user_session.get("state")
    question = message.content
    state.messages += [HumanMessage(content=question)]
    ui_message = cl.Message(content="")
    await ui_message.send()
    async for event in graph.astream_events(state, version="v2"):
        print(event)
        if event["event"] == "on_chain_end" and event["name"] == "test":
            # Try to get router from event input (contains updated state)
            # For DeepSeek and other models, output might be None, but input contains the updated state
            if event.get("data") is not None and isinstance(event["data"], dict):
                # First try to get from output (ChatGPT compatibility)
                output = event["data"].get("output")
                if output is not None and isinstance(output, dict):
                    router = output.get("router")
                    if router is not None:
                        await classification_step(router)
                        continue
                
                # Fallback: get from input (DeepSeek compatibility)
                input_data = event["data"].get("input")
                if input_data is not None and hasattr(input_data, "router"):
                    router = input_data.router
                    if router is not None:
                        await classification_step(router)

        if event["name"] == "analyze_and_route_query": 
            classification = event["data"]["output"]["router"]
            await classification_step(classification)
        
        if event["event"] == "on_chain_end":
            output = event['data'].get('output')
            if isinstance(output, dict) and "messages" in output:
                last_message = output["messages"][-1]
                if isinstance(last_message, AIMessage):
                    ui_message.content = last_message.content
    await ui_message.update()

    state.messages += [AIMessage(content=ui_message.content)]

