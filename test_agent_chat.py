import asyncio
from chat.kb_chat import agent_chat
from fastapi import Body

async def test_agent_chat():
    try:
        # 模拟 FastAPI 的 Body 参数
        query = "你好"
        top_k = 3
        score_threshold = 0.5
        kb_name = ""
        prompt_name = "default"
        model = "qwen-max"
        temperature = 0.5
        
        # 调用 agent_chat 函数
        response = await agent_chat(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            kb_name=kb_name,
            prompt_name=prompt_name,
            model=model,
            temperature=temperature
        )
        
        print("Test successful!")
        print(f"Response type: {type(response)}")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent_chat())