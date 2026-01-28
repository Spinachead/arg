from dataclasses import dataclass
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


@dataclass(kw_only=True)
class InputState:
    """
    Represents the input state containing a list of messages.

    Attributes:
        messages (list[AnyMessage]): The list of messages associated with the state, 
            processed using the add_messages function.
    """
    messages: Annotated[list[AnyMessage], add_messages]
