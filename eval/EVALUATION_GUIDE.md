# AutoResearcher 完整评测流程

本文档说明如何对 AutoResearcher 的运行结果进行完整的两阶段评测。

---

## 评测体系概览

AutoResearcher 的评测分为两个层级：

| 层级 | 评测对象 | 核心指标 | 脚本 |
|------|---------|---------|------|
| **Layer 1: RAG 检索评测** | RAG 子图的检索质量 | Event Recall, Article Recall, 双路召回贡献 | `eval_rag_retrieval.py` |
| **Layer 2: Finding 评测** | 从报告中抽取事件的准确性 | Finding Recall, Precision, F1, Evidence Support Rate | `eval_findings.py` |

---

## 前置准备

### 1. 确保 Ground Truth 数据已生成

```bash
python eval/prepare_ground_truth.py --no-db
```

**输出**：
- `eval/article_labels_v2.json`
- `eval/event_to_articles.json`
- `eval/ground_truth_stats.txt`

### 2. 确保有可评测的运行结果

运行结果目录应包含以下文件之一：
- `compressed.md`（RAG 压缩报告）或
- `report.md`（完整研究报告）

对于 RAG 检索评测，还需要：
- `retrieval_details.json`（检索详情，需要新版本代码生成）

---

## 评测流程

### 阶段 1：RAG 检索评测

**目标**：评估 RAG 子图在检索阶段的表现，分析 dense/sparse/merged/reranked 各阶段的召回率。

#### 1.1 运行评测

```bash
python eval/eval_rag_retrieval.py --run-dir logs/<your-run-dir>
```

**示例**：
```bash
python eval/eval_rag_retrieval.py --run-dir "logs/查找2026年3月1日至3月31日期间发布的大模型相关新闻产-20260507-190148"
```

#### 1.2 查看结果

评测完成后，会在运行目录下生成：

| 文件 | 说明 |
|------|------|
| `RETRIEVAL_REPORT.md` | 检索评测报告（人类可读） |
| `retrieval_metrics.json` | 检索指标（JSON 格式） |

**报告示例**：
```markdown
# RAG 检索细粒度评测报告

## 各阶段召回率

| 阶段 | 召回文章数 | Event Recall | Article Recall |
|------|-----------|--------------|----------------|
| 向量检索 (Dense) | 50 | 65.0% (13/20) | 45.0% (45/100) |
| 词法检索 (Sparse) | 45 | 55.0% (11/20) | 40.0% (40/100) |
| 合并候选池 (Merged) | 65 | 80.0% (16/20) | 60.0% (60/100) |
| Rerank Top K | 10 | 75.0% (15/20) | 10.0% (10/100) |

## 双路召回贡献分析

| 类型 | 文章数 | 命中事件数 |
|------|--------|-----------|
| 只被向量检索召回 | 20 | 5 |
| 只被词法检索召回 | 15 | 3 |
| 两路都召回 | 30 | 12 |
```

#### 1.3 关键指标解读

- **Event Recall**：有多少个 ground truth 事件被检索到
  - 高 Event Recall 说明检索覆盖面广
  - 低 Event Recall 说明漏掉了重要事件

- **Article Recall**：有多少篇相关文章被检索到
  - 高 Article Recall 说明检索全面
  - 低 Article Recall 说明检索不够深入

- **双路召回贡献**：分析向量检索和词法检索各自的贡献
  - 重叠度高：两路检索结果相似
  - 重叠度低：两路检索互补性强

#### 1.4 注意事项

⚠️ **如果运行目录缺少 `retrieval_details.json`**：

说明这是旧版本代码生成的运行结果，无法进行 RAG 检索评测。

**解决方案**：
1. 使用新版本代码重新运行 AutoResearcher
2. 或者跳过 RAG 检索评测，直接进行 Finding 评测

---

### 阶段 2：Finding 评测

**目标**：评估从研究报告中抽取事件的准确性，以及抽取的事件与 ground truth 的匹配度。

#### 2.1 运行评测

```bash
python eval/eval_findings.py --run-dir logs/<your-run-dir>
```

**示例**：
```bash
python eval/eval_findings.py --run-dir "logs/查找2026年3月1日至3月31日期间发布的大模型相关新闻产-20260507-190148"
```

#### 2.2 查看结果

评测完成后，会在运行目录下生成：

| 文件 | 说明 |
|------|------|
| `FINDING_REPORT.md` | Finding 评测报告（人类可读） |
| `finding_metrics.json` | Finding 指标（JSON 格式） |
| `finding_matches.json` | 匹配详情（JSON 格式） |

**报告示例**：
```markdown
# Finding 抽取与匹配评测报告

## 整体指标

- **Finding Recall**: 75.0% (15/20)
- **Finding Precision**: 88.2% (15/17)
- **Finding F1**: 81.1%
- **Evidence Support Rate**: 80.0%

## 正确识别的事件 (15)

### 1. OpenAI - GPT-5.4 ✓
- **匹配事件**: OpenAI GPT-5.4
- **匹配置信度**: 1.00
- **匹配原因**: 模型名称完全匹配
- **发布日期**: 2026-03-15
- **关键特性**: 支持多模态输入, 上下文窗口 200K tokens
- **证据支撑**: 3/3 篇在 gold_evidence 中

## 误报的 findings (2)

### 1. OpenAI - GPT-5.5 ✗
- **原因**: 模型名称不匹配
- **证据**: OpenAI 在3月底宣布将在4月发布 GPT-5.5...

## 漏报的事件 (5)

1. Google Gemini 2.0 ✗
2. Meta Llama 4 ✗
...
```

#### 2.3 关键指标解读

- **Finding Recall**：有多少个 ground truth 事件被正确识别
  - 高 Recall 说明报告覆盖全面
  - 低 Recall 说明报告遗漏了重要事件

- **Finding Precision**：抽取的 findings 中有多少是正确的
  - 高 Precision 说明抽取准确
  - 低 Precision 说明有误报（抽取了不存在的事件）

- **Finding F1**：Recall 和 Precision 的调和平均
  - 综合评估抽取质量

- **Evidence Support Rate**：findings 的证据支撑率
  - 高 Evidence Support Rate 说明 findings 有充分证据
  - 低 Evidence Support Rate 说明 findings 缺乏证据支撑

#### 2.4 支持的报告格式

Finding 评测支持两种报告格式：

1. **compressed.md**（RAG 压缩报告）
   - 由 RAG 子图生成
   - 通常较短，聚焦核心信息

2. **report.md**（完整研究报告）
   - 由完整 Researcher 生成
   - 通常较长，包含详细分析

脚本会自动检测并使用可用的报告文件。

---

## 完整评测示例

### 示例 1：评测已有运行结果

```bash
# 1. 准备 ground truth（如果还没有）
python eval/prepare_ground_truth.py --no-db

# 2. RAG 检索评测（如果有 retrieval_details.json）
python eval/eval_rag_retrieval.py --run-dir "logs/查找2026年3月1日至3月31日期间发布的大模型相关新闻产-20260507-190148"

# 3. Finding 评测
python eval/eval_findings.py --run-dir "logs/查找2026年3月1日至3月31日期间发布的大模型相关新闻产-20260507-190148"

# 4. 查看报告
cat "logs/查找2026年3月1日至3月31日期间发布的大模型相关新闻产-20260507-190148/RETRIEVAL_REPORT.md"
cat "logs/查找2026年3月1日至3月31日期间发布的大模型相关新闻产-20260507-190148/FINDING_REPORT.md"
```

### 示例 2：运行新查询并评测

```bash
# 1. 运行 AutoResearcher（会自动生成评测所需文件）
python -m src.main "搜索本地新闻数据库，查找2026年3月1日至3月31日期间发布的大模型相关新闻"

# 2. 找到生成的运行目录
ls -lt logs/ | head -5

# 3. 运行评测（假设运行目录是 logs/xxx）
python eval/eval_rag_retrieval.py --run-dir logs/xxx
python eval/eval_findings.py --run-dir logs/xxx
```

---

## 评测结果分析

### 如何判断系统表现好坏？

#### RAG 检索层面

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| Event Recall (Merged) | > 80% | 60-80% | < 60% |
| Article Recall (Merged) | > 70% | 50-70% | < 50% |
| Event Recall (Reranked) | > 70% | 50-70% | < 50% |

**分析要点**：
- Merged 阶段的 Event Recall 应该尽可能高（检索覆盖面）
- Reranked 阶段的 Event Recall 不应该比 Merged 低太多（重排不应该过滤掉太多相关内容）
- 双路召回应该有互补性（不是完全重叠）

#### Finding 层面

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| Finding Recall | > 75% | 60-75% | < 60% |
| Finding Precision | > 85% | 70-85% | < 70% |
| Finding F1 | > 80% | 65-80% | < 65% |
| Evidence Support Rate | > 80% | 60-80% | < 60% |

**分析要点**：
- Recall 低：报告遗漏了重要事件（可能是检索不全，或者压缩过度）
- Precision 低：报告包含了错误信息（可能是 LLM 幻觉，或者信息源不可靠）
- Evidence Support Rate 低：findings 缺乏证据支撑（可能是推测性内容过多）

---

## 常见问题

### Q1: 运行目录缺少 retrieval_details.json，怎么办？

**A**: 这是旧版本代码生成的运行结果。有两个选择：
1. 跳过 RAG 检索评测，只做 Finding 评测
2. 使用新版本代码重新运行

### Q2: Finding 评测报错 "找不到 compressed.md 或 report.md"

**A**: 运行目录不完整。确保：
- 运行已完成（不是中途中断的）
- 运行目录包含最终报告文件

### Q3: Finding Recall 很低，如何改进？

**A**: 可能的原因和改进方向：
1. **检索不全**：优化 RAG 检索策略，增加召回
2. **压缩过度**：调整压缩策略，保留更多信息
3. **LLM 抽取不准**：优化 Finding 抽取 prompt

### Q4: Finding Precision 很低，如何改进？

**A**: 可能的原因和改进方向：
1. **LLM 幻觉**：加强 prompt 约束，要求 LLM 只抽取明确提到的事件
2. **信息源不可靠**：提高检索质量，过滤低质量文章
3. **匹配策略过松**：调整匹配阈值，提高匹配精度

### Q5: 如何批量评测多个运行结果？

**A**: 可以写一个简单的 bash 脚本：

```bash
#!/bin/bash

for dir in logs/*/; do
    echo "评测: $dir"
    python eval/eval_findings.py --run-dir "$dir"
done
```

---

## 下一步

完成评测后，你可以：

1. **分析评测报告**：找出系统的优势和不足
2. **对比不同运行**：看看不同查询或配置的表现差异
3. **优化系统**：根据评测结果针对性改进
4. **持续监控**：定期评测，跟踪系统性能变化

---

## 相关文档

- [Phase 1 总结](PHASE1_SUMMARY.md) - Ground Truth 数据准备
- [Phase 2 总结](PHASE2_SUMMARY.md) - RAG 检索评测实现
- [Phase 3 总结](PHASE3_SUMMARY.md) - Finding 评测实现
- [Ground Truth 文档](README_GROUND_TRUTH.md) - Ground Truth 数据说明
