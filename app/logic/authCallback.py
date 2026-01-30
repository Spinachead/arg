import os
import chainlit as cl


async def execute(username: str, password: str):
    if (username, password) == ("admin", os.getenv("ADMIN_PASS")):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    elif (username, password) == ("user1", os.getenv("USER1_PASS")):
        return cl.User(
            identifier="user1", metadata={"role": "user", "provider": "credentials"}
        )
    return None
