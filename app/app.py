import chainlit as cl
import os
from dotenv import load_dotenv
from logic.onMessage import execute as onMessage
from logic.onChatStart import execute as onChatStart
from logic.authCallback import execute as authCallback
load_dotenv()

@cl.on_chat_start
async def start():
    await onChatStart()

@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)


@cl.on_message
async def main(message: cl.Message):
    await onMessage(message)

@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    return await authCallback(username, password)


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="GPT-3.5",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="GPT-4",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://picsum.photos/250",
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"starting chat using the {chat_profile} chat profile"
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    pass


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    from langsmith import traceable, get_current_run_tree
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")
    run_chainlit(__file__)