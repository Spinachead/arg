import chainlit as cl
import os
from dotenv import load_dotenv
from db.session import init_db
from logic.onMessage import execute as onMessage
from logic.onChatStart import execute as onChatStart
from logic.authCallback import execute as authCallback
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict

load_dotenv()

# 在启动时初始化数据库表
init_db()

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
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!", accept=["text/plain"]
        ).send()

    text_file = files[0]

    with open(text_file.path, "r", encoding="utf-8") as f:
        text = f.read()

    # Let the user know that the system is ready
    await cl.Message(
        content=f"`{text_file.name}` uploaded, it contains {len(text)} characters!"
    ).send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    from langsmith import traceable, get_current_run_tree
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")
    run_chainlit(__file__)