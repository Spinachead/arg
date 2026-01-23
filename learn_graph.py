# langgraph_rag_skeleton.py
from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
# from langgraph.store.postgres import PostgresStore  # 生产可替换

# 1) 定义状态类型
class InputState(TypedDict):
    query: str
    variants: list[str]  # query variants populated by subgraph
    retrieved_docs: list
    memory_hits: list
    final_answer: str

# 2) 子图：查询变体生成
def gen_variants_node(state: InputState):
    # 调用 LLM 或 paraphrase 模型生成变体
    variants = ["variant1 of " + state["query"], "variant2 of " + state["query"]]
    return {"variants": variants}

variants_subgraph_builder = StateGraph(InputState)
variants_subgraph_builder.add_node("gen_variants", gen_variants_node)
variants_subgraph_builder.add_edge(START, "gen_variants")
variants_subgraph = variants_subgraph_builder.compile()

# 3) 子图：检索（向量库）
def retriever_node(state: InputState):
    # 使用 state["variants"] 调用向量 DB（embedding + similarity search）
    # 返回 top-k 文档列表
    docs = [{"id": "doc1", "text": "content 1"}, {"id": "doc2", "text": "content 2"}]
    return {"retrieved_docs": docs}

retriever_subgraph_builder = StateGraph(InputState)
retriever_subgraph_builder.add_node("retrieve", retriever_node)
retriever_subgraph_builder.add_edge(START, "retrieve")
retriever_subgraph = retriever_subgraph_builder.compile()

# 4) 子图：读取用户记忆（持久化 store）
def read_memory_node(state: InputState):
    # 从 Store / memory paths 读取用户记忆
    mems = [{"id": "m1", "text": "previous preference"}]
    return {"memory_hits": mems}

memory_subgraph_builder = StateGraph(InputState)
memory_subgraph_builder.add_node("read_memory", read_memory_node)
memory_subgraph_builder.add_edge(START, "read_memory")
memory_subgraph = memory_subgraph_builder.compile()

# 5) 主图：汇总 & RAG 生成 & 写回记忆
def aggregate_node(state: InputState):
    # 将 retrieved_docs + memory_hits 合并并调用 LLM 生成 final_answer
    ctx = "\n".join([d["text"] for d in state.get("retrieved_docs", []) + state.get("memory_hits", [])])
    final = "LLM answer based on: " + ctx
    return {"final_answer": final}

def write_memory_node(state: InputState):
    # 可选：将用户明确要保存的信息写到 /memories/ 路径或 Store
    # write to persistent store here
    return {}

builder = StateGraph(InputState)
builder.add_node("variants", variants_subgraph)    # 子图作为节点
builder.add_node("retrieve", retriever_subgraph)
builder.add_node("read_memory", memory_subgraph)
builder.add_node("aggregate", aggregate_node)
builder.add_node("write_memory", write_memory_node)

# 连接：START -> variants -> parallel(retrieve, read_memory) -> aggregate -> write_memory -> END
builder.add_edge(START, "variants")
builder.add_edge("variants", "retrieve")
builder.add_edge("variants", "read_memory")
builder.add_edge("retrieve", "aggregate")
builder.add_edge("read_memory", "aggregate")
builder.add_edge("aggregate", "write_memory")

# 6) 持久化：checkpointer + store
checkpointer = MemorySaver()               # 开发时可用 MemorySaver
store = InMemoryStore()                    # 开发时使用 InMemoryStore
# 生产环境：用 PostgresStore.from_conn_string(...) 并调用 store.setup()

graph = builder.compile(checkpointer=checkpointer)  # checkpointer 会传播到子图

# 7) 运行（可流式）
config = {"configurable": {"thread_id": "thread-123"}}
for chunk in graph.stream({"query": "如何部署"}, subgraphs=True, stream_mode="updates"):
    print(chunk)

# 8) 查看状态（可选，含子图）
state = graph.get_state(config)
subgraph_state = graph.get_state(config, subgraphs=True)