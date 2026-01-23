from typing import Annotated, TypedDict



class AgentState(TypedDict):
    """ 这是graph_chat的agentState """
    messages: Annotated[list[AnyMessage], add_messages]
    knowledge: Annotated[list[dict], "知识列表"]
