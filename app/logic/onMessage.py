import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from core.state_graph.states.main_graph.router import Router


@cl.step(type="llm", name="Classify Request", show_input=False)
async def classification_step(classification: Router):
    current_step = cl.context.current_step
    await current_step.stream_token(
        f"Classified as **{classification.type}** with the logic: _{classification.logic}_"
    )


async def execute(message: cl.Message):
    graph: Runnable = cl.user_session.get("graph")
    state = cl.user_session.get("state")
    question = message.content
    state.message += [HumanMessage(content=question)]
    ui_message = cl.Message(content=="")
    await ui_message.send()
    async for event in graph.invoke(state, version="v2"):
        if event["name"] == "analyze_and_route_query": 
            classification = event["data"]["output"]["router"]
            await classification_step(classification)
    await ui_message.update()

    state.message += [AIMessage(content=ui_message.content)]

