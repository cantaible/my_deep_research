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
    # RAG execute 节点每个子查询的最大重试次数（0=不重试，只搜一次）
    max_rag_retries: int = 2

    # ── 模型配置（三层分级） ──
    #
    # Tier 1 - Simple（廉价打杂）：
    #   用于澄清判断、研究简报、压缩摘要、LATS expand/aggregate 等简单任务
    simple_model: str = "openai:gpt-4.1-mini"
    simple_model_max_tokens: int = 8192
    #
    # Tier 2 - Medium（中等难度写作）：
    #   用于最终报告生成，可挂载最擅长长文结构化写作的模型
    medium_model: str = "openai:gpt-5.4"
    medium_model_max_tokens: int = 10000
    #
    # Tier 3 - Hard（核心推理）：
    #   用于 Supervisor 决策、Researcher ReAct 工具调用、LATS evaluate 评估
    hard_model: str = "openai:gpt-5.4"
    hard_model_max_tokens: int = 10000
    #
    # ── 可独立覆盖的特化模型（默认跟随所属层级） ──
    # 摘要模型：用于汇总 Tavily 搜索结果的网页内容
    summarization_model: str = "openai:gpt-4.1-mini"
    summarization_model_max_tokens: int = 8192
    # 网页内容最大字符数，超过此长度会被截断后再摘要
    max_content_length: int = 50000
    # 压缩模型：默认跟随 simple_model；可独立覆盖
    compression_model: str = ""
    compression_model_max_tokens: int = 8192
    # 报告模型：向后兼容别名，默认跟随 medium_model
    final_report_model: str = ""
    final_report_model_max_tokens: int = 10000

    @property
    def effective_compression_model(self) -> str:
        """压缩模型：有独立配置则用，否则跟随 simple_model。"""
        return self.compression_model or self.simple_model

    @property
    def effective_final_report_model(self) -> str:
        """报告模型：有独立配置则用，否则跟随 medium_model。"""
        return self.final_report_model or self.medium_model

    # ── MCP 配置（可选，本项目暂不使用） ──
    mcp_config: Optional[MCPConfig] = None
    mcp_prompt: Optional[str] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 和环境变量创建配置实例。

        优先级：configurable 参数 > 环境变量 > 默认值。
        环境变量名 = 字段名的大写形式（如 hard_model → HARD_MODEL）。
        """
        configurable = config.get("configurable", {}) if config else {}
        # 自动遍历所有字段，不再手动一个个列举
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

