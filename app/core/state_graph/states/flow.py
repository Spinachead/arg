
from typing import Annotated, List, Dict, TypedDict, Optional, Literal
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from dataclasses import dataclass, field
from core.state_graph.states.main_graph.input_state import InputState

@dataclass
class Step:
    id: str
    description: str           # 步骤描述
    tool_name: Optional[str]   # 需要调用的工具
    tool_input: Optional[Dict] # 工具输入参数
    expected_output: str       # 预期输出
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: Optional[str] = None

@dataclass
class Plan:
    steps: List[Step] = field(default_factory=list)

@dataclass
class Observation:
    step_id: str               # 关联的步骤ID
    tool_name: Optional[str]   # 执行的工具名称
    tool_input: Optional[Dict] # 工具输入参数
    content: str               # 观察到的原始结果内容
    timestamp: str             # 观察时间戳
    status: Literal["success", "error", "timeout"] = "success"  # 执行状态
    metadata: Dict = field(default_factory=dict)  # 额外元数据（如执行耗时、结果类型等）

@dataclass  
class Reflection:
    step_id: str
    observation: str           # 观察到的结果
    is_success: bool           # 是否达到预期
    adjustment: Optional[str]  # 调整策略（如需重试）


# agent_state.py - 扩展状态定义
@dataclass(kw_only=True)
class AgentState(InputState):
    plan: Optional[Plan] = None  # 执行计划
    current_step_index: int = field(default=0)               # 当前执行步骤
    observations: List[Observation] = field(default_factory=list)  # 工具执行观察
    reflections: List[Reflection] = field(default_factory=list)    # 反思记录
    final_output: str = field(default="")                    # 最终交付物
    is_complete: bool = field(default=False)                 # 任务完成标记
    retry_count: int = field(default=0)                      # 重试计数器
    max_retries: int = field(default=3)                      # 最大重试次数






