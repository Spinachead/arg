# app.py
import json
from mcp import ClientSession
import anthropic
import chainlit as cl
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

anthropic_client = anthropic.AsyncAnthropic()

# System prompt: 告诉 Claude 我们有 memory 工具可用
SYSTEM = """
You are a helpful assistant with access to a memory system.
Users can:
- Save notes (e.g., "记住我喜欢咖啡")
- Recall notes (e.g., "我之前说过什么？")
- List all saved keys (e.g., "我记了哪些东西？")

Always use the provided tools to interact with memory.
After displaying memory content, do not repeat it — just say "here is the memory information!".
"""

# Regular tools for UI display (optional)
regular_tools = []

def flatten(xss):
    return [x for xs in xss for x in xs]

@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    """当用户在 UI 中连接 MCP 时触发"""
    result = await session.list_tools()
    tools = [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema,
        }
        for t in result.tools
    ]
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)

@cl.step(type="tool")
async def call_tool(tool_use):
    """通用工具调用处理器"""
    tool_name = tool_use.name
    tool_input = tool_use.input
    current_step = cl.context.current_step
    current_step.name = tool_name

    # 找到对应的 MCP 连接
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = None
    for connection_name, tools in mcp_tools.items():
        if any(tool.get("name") == tool_name for tool in tools):
            mcp_name = connection_name
            break

    if not mcp_name:
        current_step.output = json.dumps({"error": f"Tool {tool_name} not found"})
        return current_step.output

    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    if not mcp_session:
        current_step.output = json.dumps({"error": f"MCP {mcp_name} not connected"})
        return current_step.output

    try:
        # 调用远程 MCP 工具
        result = await mcp_session.call_tool(tool_name, tool_input)
        current_step.output = str(result)
        return current_step.output
    except Exception as e:
        current_step.output = json.dumps({"error": str(e)})
        return current_step.output

async def call_claude(chat_messages):
    """调用 Claude 并支持工具调用"""
    msg = cl.Message(content="")
    mcp_tools = cl.user_session.get("mcp_tools", {})
    tools = flatten([tools for _, tools in mcp_tools.items()])

    async with anthropic_client.messages.stream(
        system=SYSTEM,
        max_tokens=1024,
        messages=chat_messages,
        tools=tools,
        model="claude-3-5-sonnet-20240620",
    ) as stream:
        async for text in stream.text_stream:
            await msg.stream_token(text)
        await msg.send()
        response = await stream.get_final_message()
        return response

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("chat_messages", [])
    await cl.Message(
        content="🧠 Memory MCP 已就绪！你可以：\n"
                "- 说「记住我喜欢喝茶」\n"
                "- 问「我之前说了什么？」\n"
                "- 问「我记了哪些东西？」"
    ).send()

@cl.on_message
async def on_message(msg: cl.Message):
    chat_messages = cl.user_session.get("chat_messages")
    chat_messages.append({"role": "user", "content": msg.content})

    response = await call_claude(chat_messages)

    # 处理工具调用循环
    while response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")
        tool_result = await call_tool(tool_use)

        # 将工具结果反馈给模型
        messages = [
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(tool_result),
                    }
                ],
            },
        ]
        chat_messages.extend(messages)
        response = await call_claude(chat_messages)

    # 获取最终回复
    final_response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )
    if final_response:
        chat_messages.append({"role": "assistant", "content": final_response})