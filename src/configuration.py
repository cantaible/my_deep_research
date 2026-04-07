"""运行时配置。"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

for env_path in (PROJECT_ROOT / ".env", WORKSPACE_ROOT / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)


class SearchAPI(Enum):
    """可用的搜索 API 提供商。"""
    ANTHROPIC = "anthropic"  # Anthropic 原生网络搜索
    OPENAI = "openai"        # OpenAI 原生网络搜索
    TAVILY = "tavily"        # Tavily 搜索 API（默认）
    NONE = "none"            # 不使用搜索

class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) 服务器配置。
    
    MCP 允许 Agent 连接外部工具服务器，扩展可用工具集。
    本项目暂不使用，先以 stub 形式保留。
    """
    url: Optional[str] = Field(default=None, description="MCP 服务器地址")
    tools: Optional[List[str]] = Field(default=None, description="要加载的工具名列表")
    auth_required: Optional[bool] = Field(default=False, description="是否需要认证")


class Configuration(BaseModel):
    """Deep Research Agent 的主配置类。

    所有默认值与原版 open_deep_research 保持一致。
    """

    # ── 通用配置 ──
    # 结构化输出调用失败时的最大重试次数
    max_structured_output_retries: int = 3
    # 是否允许在研究前向用户提出澄清问题
    allow_clarification: bool = True
    # 最大并发研究单元数（每次 Supervisor 可同时启动的 Researcher 数量）
    max_concurrent_research_units: int = 5

    # ── 研究流程控制 ──
    # 搜索 API 选择（默认 Tavily）
    search_api: SearchAPI = SearchAPI.TAVILY
    # Supervisor 最大迭代次数（调用 ConductResearch + think_tool 的总次数上限）
    max_researcher_iterations: int = 6
    # 单个 Researcher 的最大工具调用次数（ReAct 循环上限）
    max_react_tool_calls: int = 10

    # ── 模型配置（4 个角色，各自可独立配置模型） ──
    # 摘要模型：用于汇总 Tavily 搜索结果的网页内容
    summarization_model: str = "openai:gpt-4.1-mini"
    summarization_model_max_tokens: int = 8192
    # 网页内容最大字符数，超过此长度会被截断后再摘要
    max_content_length: int = 50000
    # 研究模型：Researcher 和 Supervisor 使用的主力模型
    research_model: str = "openai:gpt-4.1"
    research_model_max_tokens: int = 10000
    # 压缩模型：将 Researcher 的搜索记录压缩为精炼摘要
    compression_model: str = "openai:gpt-4.1"
    compression_model_max_tokens: int = 8192
    # 报告模型：生成最终研究报告
    final_report_model: str = "openai:gpt-4.1"
    final_report_model_max_tokens: int = 10000

    # ── MCP 配置（可选，本项目暂不使用） ──
    mcp_config: Optional[MCPConfig] = None
    mcp_prompt: Optional[str] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 和环境变量创建配置实例。

        优先级：configurable 参数 > 环境变量 > 默认值。
        环境变量名 = 字段名的大写形式（如 research_model → RESEARCH_MODEL）。
        """
        configurable = config.get("configurable", {}) if config else {}
        # 自动遍历所有字段，不再手动一个个列举
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

