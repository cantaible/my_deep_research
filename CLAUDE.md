# Claude 工作指南

## 沟通风格

### 面试回答原则

**核心要求**：简明扼要，结构清晰，避免冗长

**回答结构**：
1. **定义**（1-2句话）
2. **核心公式/方法**（直接给出）
3. **关键点**（3-5个要点）
4. **实际结果**（数据说话）
5. **总结**（1句话）

**时长控制**：
- 简单问题：30秒-1分钟
- 复杂问题：1-2分钟
- 避免超过2分钟的回答

### 示例对比

❌ **冗长版本**：
```
让我详细解释一下...首先我们需要理解...然后...
接下来...此外...另外...还有...最后...
总的来说...综上所述...
```

✅ **简洁版本**：
```
定义：X 是衡量 Y 的指标。

计算：A / B

关键点：
- 点1
- 点2
- 点3

结果：指标值，说明问题。
```

---

## 面试回答模板

### Evidence Support Rate 如何计算

**定义**：衡量 findings 是否有高质量证据支撑。

**公式**：
```
Evidence Support Rate = 在 gold evidence 中的文章数 / 总引用文章数
```

**计算步骤**：
1. 为每个 finding 找到引用的文章（基于事件匹配）
2. 检查这些文章是否在 gold evidence 中
3. 汇总：sum(在gold中) / sum(总引用)

**实际结果**：
- 我的实现：0%
- 原因：数据结构 bug（gold_evidence 是字符串而非列表）
- 修复后预计：50-70%

**vs Faithfulness**：
- Evidence Support Rate：检查是否引用权威文章（轻量）
- Faithfulness：检查内容是否忠实于文档（需要 LLM）

---

### AutoResearcher 评测体系

**两层评测**：

**Layer 1: RAG 检索**
- 对象：Dense/Sparse/Merged/Reranked 四阶段
- 指标：Event Recall, Article Recall
- 输出：RETRIEVAL_REPORT.md

**Layer 2: Finding 评测**
- 对象：最终报告
- 指标：Finding Recall/Precision/F1, Evidence Support Rate
- 方法：LLM 抽取 + 模糊匹配
- 输出：FINDING_REPORT.md

**实际结果**：
- Finding Recall: 10% (2/20) - 严重不足
- Finding Precision: 15.4% (2/13) - 严重不足
- 问题：检索不全 + GT 不匹配

**优化方向**：
1. 提高检索召回
2. 完善 Ground Truth
3. 优化 Finding 抽取策略

---

### Ground Truth 构建

**数据**：
- 2262 篇标注文章
- 20 个事件家族

**增强**：
- 生成 canonical_name 和 aliases（规则生成，零成本）
- 构建反向索引（article_id → event_name）

**示例**：
```json
{
  "canonical_name": "GPT-5.4",
  "aliases": ["GPT 5.4", "gpt-5.4", "OpenAI GPT-5.4"]
}
```

---

## 代码风格

### 回答技术问题

**原则**：
- 先说结论，再说细节
- 用代码示例代替长篇解释
- 突出关键逻辑，省略样板代码

**示例**：

❌ **冗长**：
```
首先我们需要导入必要的库，然后定义一个函数，
这个函数接受两个参数...接下来我们需要...
```

✅ **简洁**：
```python
# 核心逻辑
def compute_rate(findings, gold_evidence):
    total = sum(len(f.evidence_ids) for f in findings)
    supported = sum(f.evidence_in_gold for f in findings)
    return supported / total if total > 0 else 0
```

---

## 文档风格

### 评测文档

**结构**：
1. 一句话概述
2. 核心指标（表格）
3. 使用方法（命令）
4. 结果示例（数据）

**避免**：
- 过长的背景介绍
- 重复的说明
- 冗余的示例

---

## 总结

**核心原则**：
- 简明 > 详细
- 结构 > 流水账
- 数据 > 描述
- 代码 > 文字

**检查清单**：
- [ ] 回答是否在2分钟内？
- [ ] 是否有清晰的结构？
- [ ] 是否用数据说话？
- [ ] 是否避免了重复？
