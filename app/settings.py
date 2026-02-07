from __future__ import annotations

import os
from pathlib import Path
import typing as t

import nltk

from pydantic_settings_file import *
from dotenv import load_dotenv
load_dotenv()

CHATCHAT_ROOT = Path(os.getenv("CHATCHAT_ROOT", ".")).resolve()



class AppSettings(BaseFileSettings):
    """应用配置 (原 config.json)"""

    model_config = SettingsConfigDict(json_file=CHATCHAT_ROOT / "app_settings.json")

    embedding_model: str = "text-embedding-3-small"
    """嵌入模型"""

    inference_model: str = "deepseek-chat"
    """推理模型名称"""

    temperature: float = 0.0
    """模型温度"""

    streaming: bool = True
    """是否启用流式输出"""

    openai_api_base: str = "https://api.deepseek.com/v1"
    """OpenAI API 基础地址"""

    openai_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    """OpenAI API 密钥"""

    max_query_length: int = 2000
    """最大查询长度"""


class BasicSettings(BaseFileSettings):
    """
    服务器基本配置信息
    """

    model_config = SettingsConfigDict(yaml_file=CHATCHAT_ROOT / "basic_settings.yaml")

    KB_ROOT_PATH: str = str(CHATCHAT_ROOT / "data/knowledge_base")
    """知识库默认存储路径"""

    SQLALCHEMY_DATABASE_URI: str = os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@127.0.0.1:5432/langchain_chatchat"
    """知识库信息数据库连接URI。默认优先从环境变量 DATABASE_URL 获取。"""

    @cached_property
    def NLTK_DATA_PATH(self) -> Path:
        """nltk 模型存储路径"""
        p = Path(__file__).parent / "data/nltk_data"
        return p


class KBSettings(BaseFileSettings):
    """知识库相关配置"""

    model_config = SettingsConfigDict(yaml_file=CHATCHAT_ROOT / "kb_settings.yaml")

    DEFAULT_KNOWLEDGE_BASE: str = "samples"
    """默认使用的知识库"""

    DEFAULT_VS_TYPE: t.Literal["faiss", "milvus", "zilliz", "pg", "es", "relyt", "chromadb"] = "faiss"
    """默认向量库/全文检索引擎类型"""

    CHUNK_SIZE: int = 750
    """知识库中单段文本长度"""

    OVERLAP_SIZE: int = 150
    """知识库中相邻文本重合长度"""

    SCORE_THRESHOLD: float = 2.0
    """知识库匹配相关度阈值，取值范围在0-2之间，SCORE越小，相关度越高"""

    ZH_TITLE_ENHANCE: bool = False
    """是否开启中文标题加强"""


class ApiModelSettings(BaseFileSettings):
    """模型配置项"""

    model_config = SettingsConfigDict(yaml_file=CHATCHAT_ROOT / "model_settings.yaml")

    DEFAULT_EMBEDDING_MODEL: str = "bge-m3"
    """默认选用的 Embedding 名称"""


class PromptSettings(BaseFileSettings):
    """Prompt 模板"""

    model_config = SettingsConfigDict(yaml_file=CHATCHAT_ROOT / "prompt_settings.yaml",
                                      json_file=CHATCHAT_ROOT / "prompt_settings.json",
                                      extra="allow")

    # 保留原有的 prompt 字典结构,因为 utils.py 中的 get_prompt_template 需要
    preprocess_model: dict = {}
    llm_model: dict = {}
    rag: dict = {
        "default": (
            "[指令] 请结合对话历史和已知信息，简洁和专业地回答问题。\n"
            "- 优先参考对话历史中的上下文信息\n"
            "- 如果已知信息中有相关内容，也请一并参考\n"
            "- 如果问题涉及之前的对话内容，请基于对话历史回答\n"
            "- 如果需要外部知识且已知信息中没有，可以说明无法从已知信息中获取\n\n"
            "[已知信息]{{context}}\n\n"
            "[问题]{{question}}\n\n"
        ),
        "empty": (
            "请你回答我的问题:\n"
            "{{question}}"
        ),
    }
    action_model: dict = {}
    postprocess_model: dict = {}


class SettingsContainer:
    CHATCHAT_ROOT = CHATCHAT_ROOT

    app_settings: AppSettings = settings_property(AppSettings())
    basic_settings: BasicSettings = settings_property(BasicSettings())
    kb_settings: KBSettings = settings_property(KBSettings())
    model_settings: ApiModelSettings = settings_property(ApiModelSettings())
    prompt_settings: PromptSettings = settings_property(PromptSettings())


Settings = SettingsContainer()
nltk.data.path.append(str(Settings.basic_settings.NLTK_DATA_PATH))
