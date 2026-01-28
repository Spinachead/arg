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

    run_chainlit(__file__)