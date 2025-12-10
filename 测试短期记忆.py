from langchain.agents import create_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def test_memory():
    llm = ChatOllama(model="qwen:1.8b", temperature=0.7)
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=llm,
        checkpointer=checkpointer,
    )
    config = RunnableConfig({"configurable": {"thread_id": "user_123"}})

    # 第一条消息
    input1 = {
        "messages": [HumanMessage(content="我的名字叫菠菜头？")]
    }
    result1 = agent.invoke(input1, config=config)
    print("问题1:", input1["messages"][0].content)
    print("回复1:", result1["messages"][-1].content)
    print()

    # 第二条消息 - 从检查点恢复之前的消息
    previous_messages = result1["messages"]  # 获取上一次的完整消息历史
    input2 = {
        "messages": previous_messages + [HumanMessage(content="我今年12岁了？")]
    }
    result2 = agent.invoke(input2, config=config)
    print("问题2:", input2["messages"][-1].content)
    print("回复2:", result2["messages"][-1].content)

    # 第三条消息 - 继续累积历史
    final_messages = result2["messages"] + [HumanMessage(content="你知道我的名字吗")]
    final_response = agent.invoke({"messages": final_messages}, config)
    final_response["messages"][-1].pretty_print()


if __name__ == "__main__":
    test_memory()