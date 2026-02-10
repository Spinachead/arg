import chainlit as cl
from core.main_graph import build_main_graph
from core.state_graph.states.main_graph.input_state import InputState
from langchain_core.messages import HumanMessage, AIMessage
from chainlit.input_widget import Select, Switch, Slider, TextInput
from db.session import session_scope
from db.repository.user_repository import get_user_settings
from settings import Settings


async def execute():
    # 初始化 graph，但不需要初始化 state
    # state 由 LangGraph 的 checkpoint 自动管理
    cl.user_session.set("graph", build_main_graph())
    
    # 加载用户保存的设置
    user = cl.user_session.get("user")

    
    saved_settings = {}
    
    if user:
        with session_scope() as session:
            saved_settings = get_user_settings(session, user.identifier)
            if saved_settings:
                # 保存到 session 中
                cl.user_session.set("model_settings", saved_settings)
    
    # 准备设置项，如果有保存的设置则使用，否则使用默认值
    model_name = saved_settings.get("model", Settings.app_settings.inference_model)
    api_key = saved_settings.get("api_key", "")
    api_base = saved_settings.get("api_base", Settings.app_settings.openai_api_base)
    temperature = saved_settings.get("temperature", Settings.app_settings.temperature)
    streaming = saved_settings.get("streaming", Settings.app_settings.streaming)
    
    # 发送 ChatSettings 到前端
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="model",
                label="模型名称",
                initial=model_name,
                description="例如: deepseek-chat, gpt-4o, qwen-turbo",
            ),
            TextInput(
                id="api_key",
                label="API Key",
                initial=api_key,
                description="你的 API 密钥，留空则使用环境变量",
            ),
            TextInput(
                id="api_base",
                label="API Base URL",
                initial=api_base,
                description="API 基础地址",
            ),
            Slider(
                id="temperature",
                label="温度 (Temperature)",
                initial=temperature,
                min=0.0,
                max=2.0,
                step=0.1,
                description="控制回答的随机性",
            ),
            Switch(
                id="streaming",
                label="流式输出",
                initial=streaming,
                description="是否启用流式输出",
            ),
        ]
    ).send()

