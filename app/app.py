import chainlit as cl
from logic.onMessage import execute as onMessage
from logic.onChatStart import execute as onChatStart


@cl.on_chat_start
async def start():
    await onChatStart()

@cl.on_message
async def main(message: cl.Message):
    await onMessage(message)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    from langsmith import traceable, get_current_run_tree
    import os
    from dotenv import load_dotenv
    load_dotenv()

    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")  # 从 LangSmith 复制
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

    run_chainlit(__file__)