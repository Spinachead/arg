import typing as t
from pydantic import BaseModel, Field, ConfigDict, computed_field
from pathlib import Path
import os

# chatchat 数据目录，必须通过环境变量设置。如未设置则自动使用当前目录。
CHATCHAT_ROOT = Path(os.environ.get("CHATCHAT_ROOT", ".")).resolve()

class MyBaseModel(BaseModel):
    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="allow",
        env_file_encoding="utf-8",
    )

class PlatformConfig(MyBaseModel):
    """模型加载平台配置"""

    platform_name: str = "xinference"
    """平台名称"""

    platform_type: t.Literal["xinference", "ollama", "oneapi", "fastchat", "openai", "custom openai"] = "xinference"
    """平台类型"""

    api_base_url: str = "http://127.0.0.1:9997/v1"
    """openai api url"""

    api_key: str = "EMPTY"
    """api key if available"""

    api_proxy: str = ""
    """API 代理"""

    api_concurrencies: int = 5
    """该平台单模型最大并发数"""

    auto_detect_model: bool = False
    """是否自动获取平台可用模型列表。设为 True 时下方不同模型类型可自动检测"""

    llm_models: t.Union[t.Literal["auto"], t.List[str]] = []
    """该平台支持的大语言模型列表，auto_detect_model 设为 True 时自动检测"""

    embed_models: t.Union[t.Literal["auto"], t.List[str]] = []
    """该平台支持的嵌入模型列表，auto_detect_model 设为 True 时自动检测"""

    text2image_models: t.Union[t.Literal["auto"], t.List[str]] = []
    """该平台支持的图像生成模型列表，auto_detect_model 设为 True 时自动检测"""

    image2text_models: t.Union[t.Literal["auto"], t.List[str]] = []
    """该平台支持的多模态模型列表，auto_detect_model 设为 True 时自动检测"""

    rerank_models: t.Union[t.Literal["auto"], t.List[str]] = []
    """该平台支持的重排模型列表，auto_detect_model 设为 True 时自动检测"""

    speech2text_models: t.Union[t.Literal["auto"], t.List[str]] = []
    """该平台支持的 STT 模型列表，auto_detect_model 设为 True 时自动检测"""

    text2speech_models: t.Union[t.Literal["auto"], t.List[str]] = []
    """该平台支持的 TTS 模型列表，auto_detect_model 设为 True 时自动检测"""


"""
   LLM模型配置，包括了不同模态初始化参数。
   `model` 如果留空则自动使用 DEFAULT_LLM_MODEL
   """

MODEL_PLATFORMS: t.List[PlatformConfig] = [
    PlatformConfig(**{
        "platform_name": "xinference",
        "platform_type": "xinference",
        "api_base_url": "http://127.0.0.1:9997/v1",
        "api_key": "EMPTY",
        "api_concurrencies": 5,
        "auto_detect_model": True,
        "llm_models": [],
        "embed_models": [],
        "text2image_models": [],
        "image2text_models": [],
        "rerank_models": [],
        "speech2text_models": [],
        "text2speech_models": [],
    }),
    PlatformConfig(**{
        "platform_name": "ollama",
        "platform_type": "ollama",
        "api_base_url": "http://127.0.0.1:11434/v1",
        "api_key": "EMPTY",
        "api_concurrencies": 5,
        "llm_models": [
            "qwen:7b",
            "qwen2:7b",
        ],
        "embed_models": [
            "quentinz/bge-large-zh-v1.5",
        ],
    }),
    PlatformConfig(**{
        "platform_name": "oneapi",
        "platform_type": "oneapi",
        "api_base_url": "http://127.0.0.1:3000/v1",
        "api_key": "sk-",
        "api_concurrencies": 5,
        "llm_models": [
            # 智谱 API
            "chatglm_pro",
            "chatglm_turbo",
            "chatglm_std",
            "chatglm_lite",
            # 千问 API
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen-max-longcontext",
            # 千帆 API
            "ERNIE-Bot",
            "ERNIE-Bot-turbo",
            "ERNIE-Bot-4",
            # 星火 API
            "SparkDesk",
        ],
        "embed_models": [
            # 千问 API
            "text-embedding-v1",
            # 千帆 API
            "Embedding-V1",
        ],
        "text2image_models": [],
        "image2text_models": [],
        "rerank_models": [],
        "speech2text_models": [],
        "text2speech_models": [],
    }),
    PlatformConfig(**{
        "platform_name": "openai",
        "platform_type": "openai",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "sk-proj-",
        "api_concurrencies": 5,
        "llm_models": [
            "gpt-4o",
            "gpt-3.5-turbo",
        ],
        "embed_models": [
            "text-embedding-3-small",
            "text-embedding-3-large",
        ],
    }),
]

XF_MODELS_TYPES = {
    "text2image": {"model_family": ["stable_diffusion"]},
    "image2image": {"model_family": ["stable_diffusion"]},
    "speech2text": {"model_family": ["whisper"]},
    "text2speech": {"model_family": ["ChatTTS"]},
}

KB_INFO: t.Dict[str, str] = {"samples": "关于本项目issue的解答"}  # TODO: 都存在数据库了，这个配置项还有必要吗？
KB_ROOT_PATH: str = str(CHATCHAT_ROOT / "data/knowledge_base")
SQLALCHEMY_DATABASE_URI: str = "sqlite:///" + str(CHATCHAT_ROOT / "data/knowledge_base/info.db")
"""知识库信息数据库连接URI"""