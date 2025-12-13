from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 创建对话模板
chat_prompt = ChatPromptTemplate([
    SystemMessage(content="你是一个友好的AI助手"),
    MessagesPlaceholder("msgs"),
    HumanMessage(content="感谢")
])


# 填充变量
formatted_messages = chat_prompt.format_messages(msgs=[HumanMessage(content="你好"), HumanMessage(content="你是谁")])

for message in formatted_messages:
    print(message.content)

# 使用ChatPromptTemplate的invoke方法
response = chat_prompt.invoke(input={"msgs": [HumanMessage(content="111")]})

# 修改这里：遍历并打印响应中的消息
for message in response.to_messages():
    print(f"消息内容: {message.content}")