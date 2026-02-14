import chainlit as cl
from core.main_graph import build_main_graph
from core.state_graph.states.main_graph.input_state import InputState
from langchain_core.messages import HumanMessage, AIMessage
from chainlit.input_widget import Select, Switch, Slider, TextInput
from db.session import session_scope
from db.repository.user_repository import get_user_settings
from settings import Settings
from db.repository.knowledge_base_repository import list_kbs_from_db
from knowledge_base.kb_api import create_kb
from logic.action import upload_document




async def execute():
    cl.user_session.set("graph", build_main_graph())
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
    knowledge_base = saved_settings.get("knowledge_base", Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE)

    existing_kbs = list_kbs_from_db()
    # 提取知识库名称字符串列表
    kb_names = [kb.kbName if hasattr(kb, 'kbName') else str(kb) for kb in existing_kbs] if existing_kbs else []
    
    # 如果没有知识库，提示用户创建
    if not kb_names:
        element = cl.CustomElement(
            name="KBConfig",
            display="inline",
            props={
                "timeout": 300,
                "title": "新建知识库",
                "description": "配置新知识库的参数",
                "fields": [
                    {"id": "kb_name", "label": "知识库名称", "type": "text", "required": True, "value": "samples"},
                    {
                        "id": "kb_info",
                        "label": "知识库简介",
                        "type": "textarea",
                        "required": False,
                        "value": "",
                        "maxLength": 300,
                        "placeholder": "用于Agent选择知识库时的描述（最多300字）",
                    },
                    {
                        "id": "embed_model",
                        "label": "嵌入模型",
                        "type": "select",
                        "options": ["text-embedding-v1", "text-embedding-v2", "text-embedding-v3"],
                        "value": "text-embedding-v1",
                        "required": True,
                    },
                    {
                        "id": "vs_type",
                        "label": "向量库类型",
                        "type": "select",
                        "options": ["faiss", "milvus", "zilliz", "pg", "es", "relyt", "chromadb"],
                        "value": "faiss",
                        "required": True,
                    },
                ],
            },
        )
        res = await cl.AskElementMessage(
            content="请配置新知识库参数:", element=element, timeout=300
        ).send()
        if res:
            result = create_kb(
                knowledge_base_name=res.get("kb_name", "samples"),
                vector_store_type=res.get("vs_type", "faiss"),
                kb_info=res.get("kb_info", ""),
                embed_model=res.get("embed_model", "text-embedding-v1"),
            )
            
            if result.code == 200:
                await cl.Message(
                    content=f"✅ 知识库 '{res.get('kb_name', 'samples')}' 创建成功！现在请上传文档。"
                ).send()
                await upload_document(res.get("kb_name", "samples"))
                # 更新知识库列表
                existing_kbs = list_kbs_from_db()
                kb_names = [kb.kbName if hasattr(kb, 'kbName') else str(kb) for kb in existing_kbs] if existing_kbs else []
            else:
                await cl.Message(
                    content=f"❌ 创建知识库失败：{result.msg}"
                ).send()
    
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
            Select(
                id="knowledge_base",
                label="知识库选择",
                values=kb_names,
                initial_value=knowledge_base if knowledge_base in kb_names else (kb_names[0] if kb_names else None),
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

