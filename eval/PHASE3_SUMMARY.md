# Phase 3 完成总结

## 已完成的工作

### 1. 定义 Finding Schema

**文件**: `eval/finding_schema.py`

**数据模型**:
- ✅ `Finding`: 从研究报告中抽取的单个事件发现
  - `event_type`: 事件类型（如 "model_release"）
  - `model_name`: 模型名称（如 "GPT-5.4"）
  - `vendor`: 厂商名称（如 "OpenAI"）
  - `release_date`: 发布日期（YYYY-MM-DD 格式）
  - `key_features`: 关键特性列表
  - `evidence_text`: 支撑该 finding 的原文摘录

- ✅ `FindingExtractionResult`: Finding 抽取结果容器
  - `findings`: 抽取的 findings 列表

- ✅ `FindingMatch`: Finding 与 ground truth 的匹配结果
  - `finding`: 原始 finding
  - `matched_event`: 匹配到的 ground truth 事件
  - `confidence`: 匹配置信度 [0, 1]
  - `match_reason`: 匹配原因说明
  - `evidence_article_ids`: 支撑文章 ID 列表
  - `evidence_in_gold`: 有多少篇在 gold_evidence 中

### 2. 实现 Finding 抽取器

**文件**: `eval/finding_extractor.py`

**功能**:
- ✅ 使用 LLM + Structured Output 从 `compressed_research` 抽取 findings
- ✅ 使用 `gpt-4o-mini` 降低成本
- ✅ 详细的 prompt 指导 LLM 识别模型发布事件
- ✅ 抽取规则：
  - 只抽取明确提到的事件，不推测
  - 同一模型只抽取一次
  - 没有明确日期时设为 null
  - evidence_text 控制在 100-200 字符

**核心函数**:
```python
async def extract_findings(compressed_research: str) -> list[Finding]
```

### 3. 实现 Finding 匹配器

**文件**: `eval/finding_matcher.py`

**匹配策略**:
- ✅ **名称标准化**: 去除空格、标点、统一大小写
- ✅ **名称相似度计算**:
  - 完全匹配: 1.0
  - 一个包含另一个: 0.8
  - 否则: 0.0
- ✅ **多维度匹配**:
  - 基于 `canonical_name` 和 `aliases` 的模糊匹配
  - 基于 `vendor` 的实体匹配
  - 综合置信度计算（阈值 0.6）

**核心函数**:
```python
def match_findings(
    findings: list[Finding],
    labels_dict: dict,
    event_index: dict,
) -> list[FindingMatch]

def compute_evidence_support(
    matches: list[FindingMatch],
    labels_dict: dict,
    event_index: dict,
) -> list[FindingMatch]
```

### 4. 实现 Finding 级别评测脚本

**文件**: `eval/eval_findings.py`

**功能**:
- ✅ 加载 ground truth 数据（`article_labels_v2.json`, `event_to_articles.json`）
- ✅ 加载压缩报告（`compressed.md`）
- ✅ 抽取 findings
- ✅ 匹配 findings 与 ground truth
- ✅ 计算评测指标：
  - **Finding Recall**: 有多少 ground truth 事件被正确识别
  - **Finding Precision**: 抽取的 findings 中有多少是正确的
  - **Finding F1**: Recall 和 Precision 的调和平均
  - **Evidence Support Rate**: findings 的证据支撑率
- ✅ 生成评测报告（`FINDING_REPORT.md`）
- ✅ 保存评测指标（`finding_metrics.json`）
- ✅ 保存匹配详情（`finding_matches.json`）

### 5. 生成文档

**文件**: `eval/PHASE3_SUMMARY.md`（本文件）

---

## 数据流

```
用户运行评测
    │
    ▼
python eval/eval_findings.py
    │
    ├─→ 运行 RAG 子图（或加载已有结果）
    │   └─→ 生成 compressed.md
    │
    ├─→ 抽取 findings
    │   └─→ finding_extractor.py
    │       └─→ LLM + Structured Output
    │           └─→ list[Finding]
    │
    ├─→ 匹配 findings
    │   └─→ finding_matcher.py
    │       ├─→ 名称相似度计算
    │       ├─→ vendor 匹配
    │       └─→ 证据支撑计算
    │           └─→ list[FindingMatch]
    │
    └─→ 计算指标 & 生成报告
        ├─→ FINDING_REPORT.md
        ├─→ finding_metrics.json
        └─→ finding_matches.json
```

---

## 输出文件

### 新增文件

| 文件 | 位置 | 说明 |
|------|------|------|
| `finding_schema.py` | `eval/` | Finding 数据模型定义 |
| `finding_extractor.py` | `eval/` | Finding 抽取器 |
| `finding_matcher.py` | `eval/` | Finding 匹配器 |
| `eval_findings.py` | `eval/` | Finding 级别评测脚本 |
| `FINDING_REPORT.md` | `logs/{run_dir}/` | Finding 评测报告 |
| `finding_metrics.json` | `logs/{run_dir}/` | 评测指标（JSON 格式） |
| `finding_matches.json` | `logs/{run_dir}/` | 匹配详情（JSON 格式） |
| `PHASE3_SUMMARY.md` | `eval/` | Phase 3 完成总结 |

### finding_matches.json 示例

```json
[
  {
    "finding": {
      "event_type": "model_release",
      "model_name": "GPT-5.4",
      "vendor": "OpenAI",
      "release_date": "2026-03-15",
      "key_features": ["支持多模态输入", "上下文窗口 200K tokens"],
      "evidence_text": "2026年3月15日，OpenAI 正式发布了 GPT-5.4..."
    },
    "matched_event": "OpenAI GPT-5.4",
    "confidence": 1.0,
    "match_reason": "模型名称完全匹配 (OpenAI GPT-5.4)",
    "evidence_article_ids": [5045, 5053, 5056],
    "evidence_in_gold": 3
  }
]
```

---

## 使用方法

### 1. 运行完整评测（包含 RAG 子图）

```bash
python eval/eval_findings.py
```

**输出**:
- 运行 RAG 子图
- 抽取 findings
- 匹配 findings 与 ground truth
- 生成报告 `logs/{run_dir}/FINDING_REPORT.md`

### 2. 只评测已有运行结果

```bash
python eval/eval_findings.py --run-dir logs/eval-xxx-xxx
```

**输出**:
- 加载已有的 `compressed.md`
- 抽取 findings
- 匹配并生成报告

### 3. 自定义查询主题

```bash
python eval/eval_findings.py --topic "搜索2026年2月的游戏相关新闻"
```

---

## 评测报告示例

```markdown
# Finding 抽取与匹配评测报告

## 查询信息
- 查询: 搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的���模型相关新闻...
- 运行目录: logs/eval-xxx-xxx

## 整体指标

- **Finding Recall**: 75.0% (15/20)
- **Finding Precision**: 88.2% (15/17)
- **Finding F1**: 81.1%
- **Evidence Support Rate**: 80.0%

## 正确识别的事件 (15)

### 1. OpenAI - GPT-5.4 ✓
- **匹配事件**: OpenAI GPT-5.4
- **匹配置信度**: 1.00
- **匹配原因**: 模型名称完全匹配 (OpenAI GPT-5.4)
- **发布日期**: 2026-03-15
- **关键特性**: 支持多模态输入, 上下文窗口 200K tokens, 推理能力提升
- **证据支撑**: 3/3 篇在 gold_evidence 中

### 2. Anthropic - Claude 4.5 ✓
- **匹配事件**: Anthropic Claude 4.5
- **匹配置信度**: 1.00
- **匹配原因**: 模型名称完全匹配 (Anthropic Claude 4.5)
- **发布日期**: 2026-03-20
- **关键特性**: 长文本理解, 代码生成, 数学推理
- **证据支撑**: 2/2 篇在 gold_evidence 中

## 误报的 findings (2)

### 1. OpenAI - GPT-5.5 ✗
- **原因**: 模型名称不匹配
- **证据**: OpenAI 在3月底宣布将在4月发布 GPT-5.5...

## 漏报的事件 (5)

1. Google Gemini 2.0 ✗
2. Meta Llama 4 ✗
3. Baidu ERNIE 5.0 ✗
4. Alibaba Qwen 3.0 ✗
5. ByteDance Doubao 2.0 ✗
```

---

## 技术亮点

### 1. LLM + Structured Output

- 使用 Pydantic 模型定义输出结构
- LLM 自动生成符合 schema 的 JSON
- 降低后处理复杂度

### 2. 鲁棒的名称匹配

- 标准化处理（去空格、标点、统一大小写）
- 支持多种名称变体（GPT-5.4 / GPT 5.4 / GPT5.4）
- 基于 aliases 的模糊匹配

### 3. 多维度置信度计算

- 名称匹配度（完全匹配 vs 部分匹配）
- vendor 匹配度
- 综合置信度阈值（0.6）

### 4. 证据追溯

- 推断哪些文章支撑了该 finding
- 检查证据是否在 gold_evidence 中
- 计算 Evidence Support Rate

### 5. 完整的评测报告

- 正确识别的事件（带详细信息）
- 误报的 findings（带原因分析）
- 漏报的事件（列表）
- 整体指标（Recall/Precision/F1/Evidence Support Rate）

---

## 技术挑战与解决方案

### 挑战 1: 抽取准确性

**问题**: LLM 可能漏掉某些事件或抽取错误信息

**解决方案**:
- 详细的 prompt 指导
- 使用 Structured Output 约束输出格式
- 要求 LLM 提供 evidence_text 以便验证

### 挑战 2: 名称变体匹配

**问题**: 同一模型有多种写法（GPT-5.4 / GPT 5.4 / gpt-5.4）

**解决方案**:
- 名称标准化（去除空格、标点、统一��小写）
- 基于 aliases 的模糊匹配
- 字符串包含关系判断

### 挑战 3: 证据追溯

**问题**: 如何从报告中反推出哪些文章支撑了哪个 finding

**解决方案**:
- 简化策略：假设所有事件文章都是证据
- 未来可以通过文本匹配（evidence_text vs 文章内容）精确推断

### 挑战 4: 边界情况

**问题**: 一个 finding 可能匹配多个 ground truth 事件

**解决方案**:
- 选择置信度最高的匹配
- 设置置信度阈值（0.6）过滤低质量匹配

---

## 下一步

Phase 3 已完成！可以开始 Phase 4（可选）：

**Phase 4: 总体结果评测**
- 整合 Phase 2（检索评测）和 Phase 3（Finding 评测）的结果
- 生成综合评测报告
- 分析检索质量对 Finding 抽取的影响

**预计工作量**: 2-3 小时

---

## 验证清单

- ✅ `finding_schema.py` 定义 Finding 数据模型
- ✅ `finding_extractor.py` 实现 Finding 抽取器
- ✅ `finding_matcher.py` 实现 Finding 匹配器
- ✅ `eval_findings.py` 实现 Finding 级别评测
- ✅ 生成评测报告和指标文件
- ⚠️ 需要实际运行测试（需要 RAG 子图运行结果）

---

## 总结

Phase 3 成功实现了 Finding 抽取与匹配评测：

1. **数据模型**: 定义了 Finding 和 FindingMatch 的结构化表示
2. **抽取器**: 使用 LLM + Structured Output 从报告中抽取事件
3. **匹配器**: 基于 aliases 和实体重叠进行鲁棒匹配
4. **评测指标**: Finding Recall/Precision/F1 + Evidence Support Rate
5. **报告生成**: 自动生成详细的 Markdown 格式评测报告

所有实现都是**模块化**的，易于扩展和维护。

---

## 三层评测体系完成情况

| 层级 | 状态 | 核心指标 |
|------|------|---------|
| **Layer 1: Researcher 检索评测** | ✅ 完成 | Event Recall, Article Recall, 双路召回贡献 |
| **Layer 2: 总体结果评测** | ✅ 完成 | Finding Recall, Finding Precision, Finding F1, Evidence Support Rate |
| **Layer 3: Agent 行为评测** | ⏸️ 暂不实施 | 查询质量、重试策略、压缩质量 |

**Phase 1 + Phase 2 + Phase 3 = 完整的两层评测体系！**
