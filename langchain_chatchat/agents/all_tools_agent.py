# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import logging
import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple, Union,
    Sequence
)

from langchain.agents import AgentExecutor
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.runnables.base import RunnableSequence
from langchain_core.tools import BaseTool
from langchain_core.utils import get_color_mapping
from langchain_core.pydantic_v1 import root_validator
from langchain_chatchat.agent_toolkits.all_tools.struct_type import (
    AdapterAllToolStructType,
)
from langchain_chatchat.agent_toolkits.mcp_kit.tools import MCPStructuredTool

from langchain_chatchat.agents.output_parsers.tools_output.drawing_tool import DrawingToolAgentAction
from langchain_chatchat.agents.output_parsers.tools_output.web_browser import WebBrowserAgentAction
from langchain_chatchat.agents.output_parsers.platform_tools import PlatformToolsAgentOutputParser
from langchain_chatchat.agents.output_parsers import MCPToolAction

logger = logging.getLogger(__name__)

NextStepOutput = List[Union[AgentFinish, MCPToolAction, AgentAction, AgentStep]]


class PlatformToolsAgentExecutor(AgentExecutor):
    mcp_tools: Sequence[MCPStructuredTool] = []

    @root_validator()
    def validate_return_direct_tool(cls, values: Dict) -> Dict:
        """Validate that tools are compatible with agent.
            TODO: platform adapter tool for all  tools,
        """
        agent = values["agent"]
        tools = values["tools"]
        if isinstance(agent.runnable, RunnableSequence):
            if isinstance(agent.runnable.last, PlatformToolsAgentOutputParser):
                for tool in tools:
                    if tool.return_direct:
                        logger.warning(
                            f"Tool {tool.name} has return_direct set to True, but it is not compatible with the "
                            f"current agent."
                        )
        # Check for multi-action agents (if applicable)
        # Note: BaseMultiActionAgent may not exist in langchain 1.0
        # This validation is kept for compatibility

        return values

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = inputs.get("intermediate_steps") if inputs.get("intermediate_steps") is not None else []
        # 确保 inputs 里不带 intermediate_steps
        if "intermediate_steps" in inputs:
            inputs = dict(inputs)
            inputs.pop("intermediate_steps")
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = inputs.get("intermediate_steps") if inputs.get("intermediate_steps") is not None else []
        # 确保 inputs 里不带 intermediate_steps
        if "intermediate_steps" in inputs:
            inputs = dict(inputs)
            inputs.pop("intermediate_steps")
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        try:
            # Use asyncio.wait_for for timeout (compatible with Python 3.7+)
            async def run_agent_loop():
                nonlocal iterations, time_elapsed
                while self._should_continue(iterations, time_elapsed):
                    next_step_output = await self._atake_next_step(
                        name_to_tool_map,
                        color_mapping,
                        inputs,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
                    if isinstance(next_step_output, AgentFinish):
                        return await self._areturn(
                            next_step_output,
                            intermediate_steps,
                            run_manager=run_manager,
                        )

                    intermediate_steps.extend(next_step_output)
                    if next_step_output:
                        # Check if all tools are special tools that should return immediately
                        special_tools = {
                            AdapterAllToolStructType.WEB_BROWSER,
                            AdapterAllToolStructType.DRAWING_TOOL,
                        }
                        regular_tools = set(name_to_tool_map.keys()) - special_tools
                        
                        # If there are regular tools, continue the loop
                        if any(action.tool in regular_tools for action, _ in next_step_output):
                            pass  # Continue with normal flow
                        else:
                            # All tools are special - return immediately
                            for next_step_action, observation in next_step_output:
                                if isinstance(next_step_action, (DrawingToolAgentAction, WebBrowserAgentAction)):
                                    tool_return = AgentFinish(
                                        return_values={"output": str(observation)},
                                        log=str(observation),
                                    )
                                    return await self._areturn(
                                        tool_return,
                                        intermediate_steps,
                                        run_manager=run_manager,
                                    )

                    if len(next_step_output) == 1:
                        next_step_action = next_step_output[0]
                        # See if tool should return directly
                        tool_return = self._get_tool_return(next_step_action)
                        if tool_return is not None:
                            return await self._areturn(
                                tool_return, intermediate_steps, run_manager=run_manager
                            )

                    iterations += 1
                    time_elapsed = time.time() - start_time
                return self.agent.return_stopped_response(
                    self.early_stopping_method, intermediate_steps, **inputs
                )
            
            output = await asyncio.wait_for(
                run_agent_loop(), timeout=self.max_execution_time
            )
            return await self._areturn(
                output, intermediate_steps, run_manager=run_manager
            )
        except (TimeoutError, asyncio.TimeoutError):
            # stop early when interrupted by the async timeout
            output = self.agent.return_stopped_response(
                self.early_stopping_method, intermediate_steps, **inputs
            )
            return await self._areturn(
                output, intermediate_steps, run_manager=run_manager
            )

    def _perform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> AgentStep:
        if run_manager:
            run_manager.on_agent_action(agent_action, color="green")
        
        if isinstance(agent_action, MCPToolAction): 
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            # Find the MCP tool by name and server_name from self.mcp_tools
            mcp_tool = None
            for tool in self.mcp_tools:
                if tool.name == agent_action.tool and tool.server_name == agent_action.server_name:
                    mcp_tool = tool
                    break
            
            if mcp_tool:
                observation = mcp_tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color="blue",
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                observation = f"MCP tool '{agent_action.tool}' from server '{agent_action.server_name}' not found in available MCP tools"

        # Otherwise we lookup the tool
        elif agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            # TODO: platform adapter tool for all  tools,
            #       view tools binding langchain_chatchat/agents/platform_tools/base.py:188
            if agent_action.tool in AdapterAllToolStructType.__members__.values():
                observation = tool.run(
                    {
                        "agent_action": agent_action,
                    },
                    verbose=self.verbose,
                    color="red",
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
        else:
            # Tool not found - return error message
            available_tools = ", ".join(list(name_to_tool_map.keys()))
            observation = (
                f"Tool '{agent_action.tool}' not found. "
                f"Available tools are: {available_tools}"
            )
        return AgentStep(action=agent_action, observation=observation)

    def _consume_next_step(
            self, values: NextStepOutput
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        if isinstance(values[-1], AgentFinish):
            return values[-1]
        else:
            return [
                (a.action, a.observation) for a in values if isinstance(a, AgentStep)
            ]

    async def _aperform_agent_action(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        agent_action: AgentAction,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AgentStep:
        if run_manager:
            await run_manager.on_agent_action(
                agent_action, verbose=self.verbose, color="green"
            )
        if isinstance(agent_action, MCPToolAction): 
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            # Find the MCP tool by name and server_name from self.mcp_tools
            mcp_tool = None
            for tool in self.mcp_tools:
                if tool.name == agent_action.tool and tool.server_name == agent_action.server_name:
                    mcp_tool = tool
                    break
            
            if mcp_tool:
                observation = await mcp_tool.arun(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color="blue",
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                observation = f"MCP tool '{agent_action.tool}' from server '{agent_action.server_name}' not found in available MCP tools"

        # Otherwise we lookup the tool
        elif agent_action.tool in name_to_tool_map:
            tool = name_to_tool_map[agent_action.tool]
            return_direct = tool.return_direct
            color = color_mapping[agent_action.tool]
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            if return_direct:
                tool_run_kwargs["llm_prefix"] = ""
            # We then call the tool on the tool input to get an observation
            # TODO: platform adapter tool for all  tools,
            #       view tools binding
            #       langchain_chatchat.agents.platform_tools.base.PlatformToolsRunnable.paser_all_tools
            if agent_action.tool in AdapterAllToolStructType.__members__.values():
                observation = await tool.arun(
                    {
                        "agent_action": agent_action,
                    },
                    verbose=self.verbose,
                    color="red",
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                observation = await tool.arun(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
        else:
            # Tool not found - return error message
            available_tools = ", ".join(list(name_to_tool_map.keys()))
            observation = (
                f"Tool '{agent_action.tool}' not found. "
                f"Available tools are: {available_tools}"
            )
        return AgentStep(action=agent_action, observation=observation)
