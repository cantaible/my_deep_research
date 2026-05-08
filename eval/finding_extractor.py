"""Finding 抽取器：从研究报告中抽取结构化的事件信息。

使用 LLM + Structured Output 从 compressed_research 中识别和抽取事件。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finding_schema import Finding, FindingExtractionResult

# 加载 .env 文件
load_dotenv(PROJECT_ROOT / ".env")


# ── Prompt ──

FINDING_EXTRACTION_PROMPT = """你是一个专业的信息抽取助手。你的任务是从研究报告中识别和抽取结构化的事件信息。

**任务说明**：
从给定的研究报告中，识别所有提到的**模型发布事件**（model_release），并为每个事件抽取以下信息：

1. **event_type**: 固定为 "model_release"
2. **model_name**: 模型名称（如 "GPT-5.4", "Claude 4.5"）
3. **vendor**: 厂商名称（如 "OpenAI", "Anthropic", "Google"）
4. **release_date**: 发布日期（格式 YYYY-MM-DD，如果报告中未明确提及则为 null）
5. **key_features**: 关键特性列表（如 ["支持多模态", "上下文窗口 200K"]）
6. **evidence_text**: 支撑该事件的原文摘录（用于后续匹配文章，尽量包含模型名称和厂商）

**抽取规则**：
- 只抽取**明确提到的模型发布事件**，不要推测或补充报告中没有的信息
- 如果同一个模型被多次提及，只抽取一次
- 如果报告中没有明确的发布日期，release_date 设为 null
- evidence_text 应该是报告中的原文片段，长度控制在 100-200 字符
- 如果报告中没有提到任何模型发布事件，返回空列表

**示例**：

输入报告：
```
2026年3月，OpenAI 发布了 GPT-5.4，这是一个支持多模态输入的大模型，上下文窗口达到 200K tokens。
同月，Anthropic 也推出了 Claude 4.5，主打长文本理解能力。
```

输出：
```json
{
  "findings": [
    {
      "event_type": "model_release",
      "model_name": "GPT-5.4",
      "vendor": "OpenAI",
      "release_date": "2026-03-01",
      "key_features": ["支持多模态输入", "上下文窗口 200K tokens"],
      "evidence_text": "2026年3月，OpenAI 发布了 GPT-5.4，这是一个支持多模态输入的大模型，上下文窗口达到 200K tokens。"
    },
    {
      "event_type": "model_release",
      "model_name": "Claude 4.5",
      "vendor": "Anthropic",
      "release_date": "2026-03-01",
      "key_features": ["长文本理解能力"],
      "evidence_text": "同月，Anthropic 也推出了 Claude 4.5，主打长文本理解能力。"
    }
  ]
}
```

现在，请从以下研究报告中抽取所有模型发布事件。
"""


# ── 抽取器 ──

def get_extractor_model():
    """获取用于 Finding 抽取的 LLM。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "未设置 OPENAI_API_KEY 环境变量\n"
            "请在项目根目录的 .env 文件中设置：\n"
            "OPENAI_API_KEY=your-api-key"
        )

    model = ChatOpenAI(
        model="gpt-4o-mini",  # 使用 mini 模型降低成本
        temperature=0,
        api_key=api_key,
    )

    return model.with_structured_output(FindingExtractionResult)


async def extract_findings(compressed_research: str) -> list[Finding]:
    """从研究报告中抽取 findings。

    Args:
        compressed_research: 压缩后的研究报告（Markdown 格式）

    Returns:
        抽取的 findings 列表
    """
    model = get_extractor_model()

    result = await model.ainvoke([
        SystemMessage(content=FINDING_EXTRACTION_PROMPT),
        HumanMessage(content=f"研究报告：\n\n{compressed_research}"),
    ])

    return result.findings


# ── 测试 ──

if __name__ == "__main__":
    import asyncio

    # 测试用例
    test_report = """
    # 2026年3月大模型发布总结

    ## OpenAI GPT-5.4
    2026年3月15日，OpenAI 正式发布了 GPT-5.4，这是 GPT 系列的最新版本。
    主要特性包括：
    - 支持多模态输入（文本、图像、音频）
    - 上下文窗口扩展到 200K tokens
    - 推理能力显著提升

    ## Anthropic Claude 4.5
    3月20日，Anthropic 推出了 Claude 4.5，主打长文本理解和分析能力。
    该模型在代码生成和数学推理方面表现优异。

    ## Google Gemini 2.0
    Google 在3月底发布了 Gemini 2.0，集成了最新的多模态技术。
    """

    async def test():
        findings = await extract_findings(test_report)
        print(f"抽取到 {len(findings)} 个 findings：\n")
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding.vendor} - {finding.model_name}")
            print(f"   发布日期: {finding.release_date}")
            print(f"   关键特性: {finding.key_features}")
            print(f"   证据: {finding.evidence_text[:80]}...")
            print()

    asyncio.run(test())
