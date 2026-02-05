# Main Graph MCP 工具集成

## 功能说明

main_graph 现已支持 MCP (Model Context Protocol) 工具调用。

## 工作流程

```
START 
  → generate_queries (生成查询)
  → retrieve_documents (检索文档)
  → response (生成回复)
  → [条件判断]
      ├─ 有工具调用 → tools (执行MCP工具) → 回到 response
      └─ 无工具调用 → END
```

## 关键组件

### 1. respond.py
- 从 Chainlit session 获取 MCP 工具
- 使用 `model.bind_tools()` 绑定工具到模型
- 模型会自动决策是否需要调用工具

### 2. mcp_tool_node.py
- 处理工具调用请求
- 查找对应的 MCP 连接并执行工具
- 返回工具执行结果
- 使用 Chainlit Step 显示工具调用过程

### 3. main_graph.py
- 添加 `tools` 节点
- 使用 `should_continue` 判断是否需要调用工具
- 工具执行后循环回 response 节点

## 使用示例

用户可以在聊天中直接请求使用 MCP 工具，例如:
- "记住我喜欢喝咖啡" (memory 工具)
- "我之前说过什么?" (memory 工具)
- 其他已连接的 MCP 工具

模型会自动判断是否需要调用工具，并执行相应操作。

## 注意事项

1. 确保在 app.py 中正确配置了 `@cl.on_mcp_connect` 回调
2. MCP 工具需要在聊天开始前连接
3. 工具调用过程会在 Chainlit UI 中显示为独立的 step
