import chainlit as cl
import os
import io
import asyncio
from dotenv import load_dotenv
from settings import Settings
from db.session import init_db
from logic.onMessage import execute as onMessage
from logic.onChatStart import execute as onChatStart
from logic.authCallback import execute as authCallback
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from logic.action import upload_document
from knowledge_base.kb_api import init_default_kb
from chainlit.input_widget import Select, Switch, Slider,TextInput
from mcp import ClientSession


load_dotenv()

# 在启动时初始化数据库表
asyncio.run(init_db())
# init_default_kb()

@cl.on_chat_start
async def start():
    await onChatStart()
    
@cl.set_starters
async def set_starters():
    return [
         cl.Starter(
            label="新建知识库",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/cat.png",
        ),
         cl.Starter(
            label="管理知识库",
            message="Can you help me create a personalized morning routine that would help increase my productivity throughout the day? Start by asking me about my current habits and what activities energize me in the morning.",
            icon="/public/cat.png",
        ),
    ]

@cl.on_message
async def main(message: cl.Message):
    await onMessage(message)

@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    return await authCallback(username, password)

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo=os.getenv("DATABASE_URL"),
    )

@cl.action_callback("action_button")
async def on_action(action: cl.Action):
    print(action.payload)

@cl.on_chat_resume
async def on_resume(thread: ThreadDict):
    await onChatStart()

@cl.action_callback("upload_document")
async def on_action(action: cl.Action):
    await upload_document()


@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established"""
    # Your connection initialization code here
    # This handler is required for MCP to work
    result = await session.list_tools()
    tools = [{
        "name": t.name,
        "description": t.description,
        "input_schema": t.inputSchema,
        } for t in result.tools]
    
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)
    
@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    # Your cleanup code here
    # This handler is optional

@cl.on_settings_update
async def setup_agent(settings):
    """当用户更新设置时保存到数据库"""
    print("on_settings_update", settings)
    
    from db.session import session_scope
    from db.repository.user_repository import save_user_settings
    
    # 获取当前用户
    user = cl.user_session.get("user")
    if not user:
        print("用户未登录，无法保存设置")
        return
    
    # 保存设置到数据库
    with session_scope() as session:
        success = save_user_settings(session, user.identifier, settings)
        if success:
            print(f"用户 {user.identifier} 的设置已保存")
            
            # 同时保存到 session 中，供当前会话使用
            cl.user_session.set("model_settings", settings)
        else:
            print(f"保存用户设置失败")

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    from langsmith import traceable, get_current_run_tree
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")
    run_chainlit(__file__)