"""Finding 数据模型定义。

Finding 是从研究报告中抽取的结构化事件信息。
"""

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """从研究报告中抽取的单个事件发现。"""

    event_type: str = Field(
        description="事件类型，如 'model_release', 'product_launch', 'acquisition' 等"
    )

    model_name: str = Field(
        description="模型或产品名称，如 'GPT-5.4', 'Claude 4.5' 等"
    )

    vendor: str = Field(
        description="厂商或公司名称，如 'OpenAI', 'Anthropic', 'Google' 等"
    )

    release_date: str | None = Field(
        default=None,
        description="发布日期，格式为 YYYY-MM-DD，如果报告中未明确提及则为 None"
    )

    key_features: list[str] = Field(
        default_factory=list,
        description="关键特性或亮点列表"
    )

    evidence_text: str = Field(
        default="",
        description="支撑该 finding 的原文摘录（用于后续匹配文章）"
    )


class FindingExtractionResult(BaseModel):
    """Finding 抽取结果。"""

    findings: list[Finding] = Field(
        description="从研究报告中抽取的所有 findings"
    )


class FindingMatch(BaseModel):
    """Finding 与 ground truth 的匹配结果。"""

    finding: Finding
    matched_event: str | None = Field(
        default=None,
        description="匹配到的 ground truth 事件的 canonical_name，如果未匹配则为 None"
    )
    confidence: float = Field(
        default=0.0,
        description="匹配置信度，范围 [0, 1]"
    )
    match_reason: str = Field(
        default="",
        description="匹配原因说明"
    )
    evidence_article_ids: list[int] = Field(
        default_factory=list,
        description="支撑该 finding 的文章 ID 列表（通过文本匹配推断）"
    )
    evidence_in_gold: int = Field(
        default=0,
        description="有多少篇证据文章在 gold_evidence 中"
    )
