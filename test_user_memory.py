"""
测试用户长期记忆功能
"""
from db.repository.user_memory_repository import (
    add_user_memory,
    list_user_memories,
    get_user_profile_from_memories,
    create_memory_from_conversation,
    update_memory_last_used,
)
from utils import build_logger

logger = build_logger()


def test_user_memory():
    """测试用户记忆功能"""
    
    # 测试用户ID（确保这个用户在数据库中存在）
    test_user_id = 1
    
    print("=" * 50)
    print("1. 创建用户记忆")
    print("=" * 50)
    
    # 添加几条测试记忆
    memory_id_1 = add_user_memory(
        user_id=test_user_id,
        memory_text="用户经常询问法律合规相关问题",
        importance=5,
        meta_data={
            "kb_name": "law_kb",
            "domains": ["law", "compliance"],
            "tone": "专业详细"
        }
    )
    print(f"创建记忆 1: ID={memory_id_1}")
    
    memory_id_2 = create_memory_from_conversation(
        user_id=test_user_id,
        conversation_summary="用户询问公司内部产品文档和API使用方法",
        kb_name="company_docs",
        domains=["product", "api"],
        importance=4
    )
    print(f"创建记忆 2: ID={memory_id_2}")
    
    memory_id_3 = add_user_memory(
        user_id=test_user_id,
        memory_text="用户偏好简洁明了的回答风格",
        importance=3,
        meta_data={
            "tone": "简明扼要"
        }
    )
    print(f"创建记忆 3: ID={memory_id_3}")
    
    print("\n" + "=" * 50)
    print("2. 获取用户记忆列表")
    print("=" * 50)
    
    memories = list_user_memories(user_id=test_user_id, limit=10)
    for memory in memories:
        print(f"\nID: {memory.id}")
        print(f"内容: {memory.memory_text}")
        print(f"重要程度: {memory.importance}")
        print(f"元数据: {memory.meta_data}")
        print(f"创建时间: {memory.create_time}")
    
    print("\n" + "=" * 50)
    print("3. 获取用户画像")
    print("=" * 50)
    
    profile = get_user_profile_from_memories(user_id=test_user_id)
    print(f"用户画像:")
    print(f"  偏好知识库: {profile['preferred_kbs']}")
    print(f"  偏好领域: {profile['preferred_domains']}")
    print(f"  回答风格: {profile['tone']}")
    print(f"  最近话题: {profile['recent_topics'][:3]}")
    
    print("\n" + "=" * 50)
    print("4. 更新记忆使用时间")
    print("=" * 50)
    
    success = update_memory_last_used(memory_id=memory_id_1)
    print(f"更新记忆 {memory_id_1} 使用时间: {'成功' if success else '失败'}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    try:
        test_user_memory()
    except Exception as e:
        logger.exception(f"测试失败: {e}")
