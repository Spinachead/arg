import chainlit as cl
import os
import io
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

load_dotenv()

# 在启动时初始化数据库表
init_db()
init_default_kb()

@cl.on_chat_start
async def start():
    await onChatStart()

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

from mcp import ClientSession

@cl.on_mcp_connect
async def on_mcp_connect(connection, session: ClientSession):
    """Called when an MCP connection is established"""
    # Your connection initialization code here
    # This handler is required for MCP to work
    
@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session: ClientSession):
    """Called when an MCP connection is terminated"""
    # Your cleanup code here
    # This handler is optional


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    from langsmith import traceable, get_current_run_tree
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")
    run_chainlit(__file__)