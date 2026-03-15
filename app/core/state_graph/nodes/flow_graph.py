import asyncio
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from settings import Settings
from core.prompts import *
from typing import cast
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from db.db_schema import DB_SCHEMA
from core.state_graph.states.flow import *
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from core.state_graph.knowledge_query_graph import knowledge_graph
from typing import Any
from core.state_graph.states.main_graph.router import Router



async def analyze_and_route_query(state: AgentState, *, config: RunnableConfig) -> dict[str, Router]:
    """
    分析当前代理状态并确定下一步的route logic
    """
    model = init_chat_model(
        name="analyze_and_route_query",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )
    struct_model = model.with_structured_output(Router)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
    ])
    chain = chat_prompt | struct_model
    response = cast(
        Router, await chain.ainvoke({
            "history": state.messages,
            "DB_SCHEMA": DB_SCHEMA
        })
    )
    print(f"\033[92mUsing analyze_and_route_query: {response}\033[0m")  # 绿色输出
    return {"router": response}


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """
    根据当前的route logic向用户询问更多信息。
    """

    model = init_chat_model(
        name="ask_for_more_info",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )
    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(logic=state.router.logic)
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages, config)
    return {"messages": [response]}

#这个方法要在create_plan中调用
async def review_plan(plan: Plan) -> Plan:
    """ 审查研究计划以确保其质量和相关性"""

    formatted_plan = ""
    for i, step in enumerate(plan["steps"]):
        formatted_plan += f"{i+1}. ({step['type']}): {step['question']}\n"
    model = init_chat_model(
        name="planner",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )
    struct_model = model.with_structured_output(Plan)
    chat_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("history"),
    ])
    chain = chat_prompt | struct_model
    reviewed_plan = cast(
        Plan, await chain.ainvoke({
        })
    )
    return reviewed_plan

async def create_plan(state: AgentState, *, config: RunnableConfig) -> dict:
    """ 根据用户的问题生成计划"""
    model = init_chat_model(
        name="planner",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )
    struct_model = model.with_structured_output(Plan)
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", PLAN_PROMPT),
    ])
    chain = chat_prompt | struct_model

    response = cast(
        Plan, await chain.ainvoke({
            "history": state.messages,
        })
    )
    print(f"\033[92mUsing create_plan: {response}\033[0m")
    return {"plan": response["steps"]}


async def conduct_knowledge(state: AgentState) -> dict[str, Any]:
    """
    执行 knowledge_graph 子图节点
    """
    
    # 调用知识查询子图
    response = await knowledge_graph.ainvoke({
        "messages": state.messages,
    })

    print(f"\033[92mUsing conduct_knowledge: {response}\033[0m")  # 绿色输出
    # 将子图的 context 返回给主图状态
    return {
        "context": response.get("context", ""),
    }


# nodes/agent/executor.py
async def executor(state: AgentState, config: RunnableConfig) -> Dict:
    """执行当前步骤，准备工具调用"""
    plan = state.plan
    current_idx = state.current_step_index
    
    if current_idx >= len(plan):
        return {"is_complete": True}
    
    step = plan[current_idx]
    
    # 检查依赖是否完成
    for dep_id in step.dependencies:
        dep_step = next((s for s in plan if s.id == dep_id), None)
        if not dep_step or dep_step.status != "completed":
            return {
                "messages": [AIMessage(content=f"等待依赖步骤 {dep_id} 完成...")],
            }
    
    # 更新步骤状态
    step.status = "running"
    
    # 如果有工具需要调用，准备工具调用消息
    if step.tool_name:
        tool_call = {
            "id": f"call_{step.id}",
            "name": step.tool_name,
            "args": resolve_tool_input(step.tool_input, state.observations),
        }
        return {
            "messages": [AIMessage(content="", tool_calls=[tool_call])],
        }
    else:
        # 纯推理步骤，直接执行
        result = await execute_reasoning_step(step, state)
        step.status = "completed"
        step.result = result
        return {
            "messages": [AIMessage(content=result)],
            "current_step_index": current_idx + 1,
        }

# 反思的节点
async def reflector(state: AgentState, config: RunnableConfig) -> Dict:
    """反思上一步执行结果，决定下一步行动"""
    plan = state.plan
    current_idx = state.current_step_index
    step = plan[current_idx]
    
    # 获取最新的观察结果
    last_observation = state.observations[-1] if state.observations else None
    
    model = init_chat_model(temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([("system", REFLECTOR_PROMPT)])
    chain = prompt | model | JsonOutputParser()
    
    reflection_data = await chain.ainvoke({
        "step_id": step.id,
        "step_description": step.description,
        "expected_output": step.expected_output,
        "actual_result": last_observation.content if last_observation else "无结果",
    })
    
    reflection = Reflection(
        step_id=step.id,
        observation=reflection_data["assessment"],
        is_success=reflection_data["is_success"],
        adjustment=reflection_data.get("adjustment_plan"),
    )
    
    # 根据反思结果决策
    action = reflection_data["action"]
    
    if action == "continue":
        step.status = "completed"
        step.result = last_observation.content if last_observation else ""
        return {
            "reflections": [reflection],
            "current_step_index": current_idx + 1,
            "retry_count": 0,
        }
    elif action == "retry":
        if state.retry_count >= state.max_retries:
            return {
                "reflections": [reflection],
                "messages": [AIMessage(content=f"步骤 {step.id} 重试次数超限，跳过")],
                "current_step_index": current_idx + 1,
                "retry_count": 0,
            }
        step.status = "pending"
        return {
            "reflections": [reflection],
            "retry_count": state.retry_count + 1,
        }
    elif action == "adjust":
        # 动态调整当前步骤
        step.tool_input = adjust_tool_input(step, reflection.adjustment)
        step.status = "pending"
        return {
            "reflections": [reflection],
            "retry_count": state.retry_count + 1,
        }


# 整合交付的节点
async def integrator(state: AgentState, config: RunnableConfig) -> Dict:
    """整合所有步骤结果，生成最终输出"""
    
    # 收集所有成功步骤的结果
    completed_steps = [s for s in state.plan if s.status == "completed"]
    
    model = init_chat_model(temperature=0.5)
    
    integration_prompt = """基于以下执行结果，生成完整的最终答案：

任务目标: {goal}

执行步骤及结果:
{step_results}

反思记录:
{reflections}

请生成:
1. 执行摘要
2. 详细结果
3. 如有交付物，提供完整内容
"""
    
    chain = ChatPromptTemplate.from_messages([("system", integration_prompt)]) | model
    
    result = await chain.ainvoke({
        "goal": state.plan[0].description if state.plan else "",
        "step_results": "\n".join([
            f"- {s.id}: {s.description}\n  结果: {s.result}" 
            for s in completed_steps
        ]),
        "reflections": "\n".join([
            f"- {r.step_id}: {r.observation}" 
            for r in state.reflections
        ]),
    })
    
    return {
        "final_output": result.content,
        "messages": [result],
    }

async def deliver(state: AgentState, config: RunnableConfig) -> Dict:
    """根据任务类型执行最终交付"""
    
    # 检查是否需要发送邮件
    if "邮件" in state.messages[0].content or "发送" in state.messages[0].content:
        email_result = await send_email_tool.ainvoke({
            "to": state.recipients or ["team@company.com"],
            "subject": state.email_subject or "任务执行结果",
            "body": state.final_output,
        })
        return {
            "messages": [AIMessage(content=f"已发送邮件: {email_result}")],
        }
    
    # 其他交付方式...
    
    return {
        "messages": [AIMessage(content=state.final_output)],
    }

from langgraph.graph import END, START, StateGraph


def build_main_graph():
    builder = StateGraph(AgentState, input=InputState)
    builder.add_node("create_plan", create_plan)
    builder.add_edge(START, "create_plan")
    builder.add_edge("create_plan", END)
    return builder.compile()


if __name__ == "__main__":
    async def main():
        main_graph = build_main_graph()
        result = await main_graph.ainvoke(input={"messages": [HumanMessage(content="如何创建一个MySQL数据库？")]})
        print(result)
    
    asyncio.run(main())
    





