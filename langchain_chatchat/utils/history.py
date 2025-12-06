# -*- coding: utf-8 -*-
import logging
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Union

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from pydantic import BaseModel, Field  # v1.0: pydantic 替换 pydantic_v1

logger = logging.getLogger()


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class History(BaseModel):
    """
    对话历史
    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msg_tuple = ("human", "你好")
    """

    role: str = Field(..., description="消息角色: human/ai/user/assistant/system")
    content: str = Field(..., description="消息内容")

    def to_msg_tuple(self):
        """转换为 (role, content) 元组"""
        return "ai" if self.role == "assistant" else "human", self.content

    def to_langchain_message(self) -> BaseMessage:
        """转换为 LangChain 标准消息对象 (v1.0 推荐)"""
        role_map = {
            "human": HumanMessage,
            "user": HumanMessage,
            "ai": AIMessage,
            "assistant": AIMessage,
            "system": SystemMessage,
        }

        MessageClass = role_map.get(self.role, ChatMessage)
        return MessageClass(content=self.content)

    def to_messages(self, history_list: List["History"]) -> List[BaseMessage]:
        """将历史列表转换为 LangChain 消息列表"""
        return [h.to_langchain_message() for h in history_list]

    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        """从数据创建 History"""
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)
        return h

    @classmethod
    def from_message(cls, message: BaseMessage) -> "History":
        """从 LangChain 消息创建 History"""
        return cls.from_data(_convert_message_to_dict(message=message))

    @classmethod
    def from_messages(cls, messages: List[BaseMessage]) -> List["History"]:
        """从消息列表创建 History 列表"""
        return [cls.from_message(msg) for msg in messages]


# 使用示例 (v1.0 风格)
def example_usage():
    """v1.0 使用示例"""

    # 1. 创建历史
    history_data = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
    ]

    # 2. 转换为 History 对象
    histories = [History.from_data(h) for h in history_data]

    # 3. 转换为 LangChain 消息列表 (推荐)
    messages = History.from_messages([
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么可以帮助你的？")
    ])

    # 4. 直接使用消息列表调用模型
    # Note: init_chat_model is now in langchain package
    # from langchain import init_chat_model
    # model = init_chat_model("gpt-4o-mini")
    # result = model.invoke(messages)
    # print(result.content)


# 兼容旧版 to_msg_template 方法 (如果需要 PromptTemplate)
def create_prompt_from_history(histories: List[History], system_prompt: str = "") -> List[BaseMessage]:
    """
    v1.0 替代 ChatMessagePromptTemplate 的方法
    将历史转换为标准消息列表
    """
    messages = []

    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    for history in histories:
        messages.append(history.to_langchain_message())

    return messages


if __name__ == "__main__":
    example_usage()