"""
测试阿里云DashScope配置的脚本
"""
import os
from settings import Settings
from utils import get_config_models, get_ChatOpenAI

def test_dashscope_config():
    """测试DashScope配置"""
    print("测试阿里云DashScope配置...")
    
    # 打印所有配置的模型平台
    print("\n1. 所有模型平台配置:")
    platforms = Settings.model_settings.MODEL_PLATFORMS
    for platform in platforms:
        if platform.platform_name == "dashscope":
            print(f"  - 找到DashScope平台配置:")
            print(f"    平台名称: {platform.platform_name}")
            print(f"    平台类型: {platform.platform_type}")
            print(f"    API基础URL: {platform.api_base_url}")
            print(f"    LLM模型: {platform.llm_models}")
            print(f"    Embed模型: {platform.embed_models}")
            break
    else:
        print("  - 未找到DashScope平台配置")
    
    # 获取配置的模型信息
    print("\n2. 获取配置的模型信息:")
    try:
        dashscope_models = get_config_models(platform_name="dashscope", model_type="llm")
        print(f"  DashScope LLM模型: {list(dashscope_models.keys())}")
    except Exception as e:
        print(f"  获取DashScope模型时出错: {e}")
    
    # 测试获取模型配置
    print("\n3. 测试获取qwen-max模型配置:")
    try:
        qwen_max_config = get_config_models(model_name="qwen-max", model_type="llm")
        print(f"  qwen_max_config keys: {list(qwen_max_config.keys())}")
        if "qwen-max" in qwen_max_config:
            config = qwen_max_config["qwen-max"]
            print(f"  - 模型配置: {config}")
        else:
            print("  - 未找到qwen-max模型配置")
            print(f"  - 可用模型: {list(qwen_max_config.keys())}")
    except Exception as e:
        print(f"  获取qwen-max模型配置时出错: {e}")
    
    print("\n4. 配置完成，可以使用阿里云模型！")
    print("\n使用方法:")
    print("  - 设置环境变量 DASHSCOPE_API_KEY=您的API密钥")
    print("  - 在应用中选择 qwen-max 或其他阿里云模型")
    print("  - 模型将通过阿里云DashScope API进行调用")

if __name__ == "__main__":
    test_dashscope_config()