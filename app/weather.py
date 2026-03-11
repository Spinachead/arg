from typing import Annotated, TypedDict, Literal
from typing_extensions import TypedDict
import operator
import os
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_deepseek import ChatDeepSeek  # 使用 DeepSeek 模型

# 加载 .env 文件
load_dotenv()

# === 工具 ===
@tool
def get_current_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F, sunny."

# === 子图状态（共享 messages，专用字段） ===
class WeatherSubgraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 共享主图消息
    subgraph_results: Annotated[list[str], operator.add]  # 收集工具结果

def weather_router(state: WeatherSubgraphState) -> Literal["agent", "execute_tools", END]:
    """路由逻辑：检查最后一条消息是否有工具调用"""
    last_msg = state["messages"][-1]
    # 如果有 tool_calls，去执行工具
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "execute_tools"
    # 如果用户提到了 weather，去 agent 节点
    if "weather" in last_msg.content.lower():
        return "agent"
    return END

def agent_node(state: WeatherSubgraphState):
    """调用模型生成工具调用"""
    model = ChatDeepSeek(model="deepseek-chat")
    tools = [get_current_weather]
    bound_model = model.bind_tools(tools)
    
    msg = bound_model.invoke(state["messages"])
    return {"messages": [msg]}

def execute_tools(state: WeatherSubgraphState):
    """执行工具调用并收集结果"""
    last_msg = state["messages"][-1]
    tool_results = []
    tool_messages = []
    
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        for tool_call in last_msg.tool_calls:
            if tool_call['name'] == 'get_current_weather':
                city = tool_call['args']['city']
                result = get_current_weather.invoke({'city': city})
                tool_results.append(result)
                tool_messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call['id']
                ))
    
    return {
        "messages": tool_messages,
        "subgraph_results": tool_results
    }

# 子图：独立 checkpointer 隔离状态
checkpointer_sub = MemorySaver()
subgraph_builder = StateGraph(WeatherSubgraphState)
subgraph_builder.add_node("agent", agent_node)
subgraph_builder.add_node("execute_tools", execute_tools)
subgraph_builder.add_conditional_edges(START, weather_router, {"agent": "agent", "execute_tools": "execute_tools", END: END})
subgraph_builder.add_conditional_edges("agent", weather_router, {"agent": "agent", "execute_tools": "execute_tools", END: END})
subgraph_builder.add_edge("execute_tools", END)
weather_subgraph = subgraph_builder.compile(checkpointer=checkpointer_sub)

# === 主图状态 ===
class MainState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 共享，所有图可见
    query_type: str  # "weather" 或其他
    subgraph_results: Annotated[list[str], operator.add]  # 收集子图输出
    final_summary: str

def route_query(state: MainState) -> Literal["weather_subgraph", END]:
    query = state["messages"][-1].content.lower()
    if "weather" in query:
        return "weather_subgraph"
    return END

def summarize_results(state: MainState) -> dict:
    results = "\n".join(state["subgraph_results"])
    return {
        "final_summary": f"Results: {results}",
        "messages": [AIMessage(content=f"Summary: {results}")]
    }

# 主图：调用子图作为节点
checkpointer_main = MemorySaver()  # 独立 checkpointer
main_builder = StateGraph(MainState)
main_builder.add_node("weather_subgraph", weather_subgraph)  # 子图作为节点
main_builder.add_node("summarize", summarize_results)
main_builder.add_conditional_edges(START, route_query, {
    "weather_subgraph": "weather_subgraph",
    END: "summarize"
})
main_builder.add_edge("weather_subgraph", "summarize")
main_builder.add_edge("summarize", END)

main_graph = main_builder.compile(checkpointer=checkpointer_main)

# === 执行 ===
config = {"configurable": {"thread_id": "example"}}
input_state = {
    "messages": [{"role": "user", "content": "What's the weather in SF and NYC?"}],
    "query_type": "multi_city",
    "subgraph_results": []
}

result = main_graph.invoke(input_state, config)
# print(f"这是result: {result}")
print(result["final_summary"])