# Ground Truth 数据准备

## 概述

`prepare_ground_truth.py` 脚本用于生成扩展后的 ground truth 数据，为评测系统提供基础数据。

## 功能

1. **canonical_name 映射**：从 `event_families.json` 自动映射事件标准名称
2. **aliases 生成**：基于规则生成事件名称的常见变体（用于模糊匹配）
3. **gold_evidence 提取**：从数据库提取关键证据片段（标题 + 正文前 300 字）
4. **反向索引**：生成 `article_id → canonical_name` 的快速查询索引

## 使用方法

### 基本用法（不连接数据库）

```bash
python eval/prepare_ground_truth.py --no-db
```

**说明**：
- 使用文章标题作为 `gold_evidence`
- 适合快速生成数据，不需要数据库连接
- 推荐用于开发和测试

### 完整用法（连接数据库）

```bash
python eval/prepare_ground_truth.py
```

**说明**：
- 从 MariaDB 数据库提取完整的 `gold_evidence`（标题 + 正文前 300 字）
- 需要数据库连接（配置在 `rag/config.py`）
- 推荐用于生产环境

## 输入文件

| 文件 | 说明 |
|------|------|
| `eval/article_labels.json` | 原始文章标注（必需） |
| `eval/event_families.json` | 事件家族定义（必需） |
| MariaDB 数据库 | 文章全文（可选，用于提取完整 evidence） |

## 输出文件

| 文件 | 说明 |
|------|------|
| `eval/article_labels_v2.json` | 扩展后的文章标注 |
| `eval/event_to_articles.json` | 反向索引（article_id → canonical_name） |
| `eval/ground_truth_stats.txt` | 统计报告 |

## 输出数据格式

### article_labels_v2.json

```json
{
  "article_id": 4362,
  "title": "DeepSeek V4 即将上线！全新多模态模型将颠覆AI界",
  "published_at": "2026-03-02",
  "source_name": "AI hot",
  "event_type": "model_release",
  "entities": ["DeepSeek", "DeepSeek V4"],
  
  // ── 新增字段 ──
  "canonical_name": "DeepSeek V4",
  "aliases": [
    "DeepSeek V4",
    "DeepSeekV4",
    "DeepSeek 4",
    "deepseek v4",
    "深度求索 DeepSeek V4"
  ],
  "gold_evidence": "DeepSeek V4 即将上线！全新多模态模型将颠覆AI界\nDeepSeek 宣布即将推出 V4 版本，这是一个全新的多模态大模型..."
}
```

### event_to_articles.json

```json
{
  "4362": "DeepSeek V4",
  "5085": "DeepSeek V4",
  "6304": "DeepSeek V4",
  "4574": "DeepSeek V4",
  "5045": "GPT-5.4",
  "5053": "GPT-5.4",
  ...
}
```

## Aliases 生成规则

脚本使用以下规则自动生成事件名称的常见变体：

1. **原始名称**：`GPT-5.4`
2. **去除连字符**：`GPT54`
3. **连字符改空格**：`GPT 5.4`
4. **小写**：`gpt-5.4`, `gpt54`, `gpt 5.4`
5. **添加厂商前缀**：
   - GPT → `OpenAI GPT-5.4`
   - Qwen → `阿里 Qwen3.5`, `通义 Qwen3.5`
   - DeepSeek → `深度求索 DeepSeek V4`
   - GLM → `智谱 GLM-5-Turbo`
   - Gemini → `Google Gemini 3.1`
   - Grok → `xAI Grok 4.20`
   - 等等...

## 统计报告示例

```
Ground Truth 数据统计
============================================================

总文章数: 2262
  有 canonical_name: 100 (4.4%)
  无 canonical_name: 2162 (95.6%)

Gold Evidence 提取:
  来源: title_only
  成功: 2262
  失败: 0

事件类型分布:
  other                  799 (35.3%)
  product_launch         532 (23.5%)
  industry_news          521 (23.0%)
  funding_business       310 (13.7%)
  model_release          100 (4.4%)

事件家族统计:
  事件家族数: 20
  覆盖文章数: 100

事件家族列表:
  GPT-5.4                         17 篇
  Qwen3.5                         11 篇
  GPT-5.3 Instant                 11 篇
  Composer                         9 篇
  MiMo                             8 篇
  ...
```

## 常见问题

### Q: 为什么只有 4.4% 的文章有 canonical_name？

A: 因为 `event_families.json` 只定义了 20 个模型发布事件（100 篇文章）。其他文章（product_launch、industry_news 等）没有归入事件家族。评测时主要关注 `model_release` 类型的文章。

### Q: gold_evidence 使用标题还是完整正文？

A: 
- **--no-db 模式**：只使用标题（快速，但信息较少）
- **数据库模式**：标题 + 正文前 300 字（推荐，信息更完整）

### Q: 如何添加新的事件家族？

A: 编辑 `eval/event_families.json`，添加新的事件和对应的 article_ids，然后重新运行脚本。

### Q: aliases 不够准确怎么办？

A: 可以修改 `generate_aliases()` 函数，添加更多规则。或者在生成后手动编辑 `article_labels_v2.json`。

## 下一步

生成 ground truth 数据后，可以：

1. **运行 RAG 检索评测**：`python eval/eval_rag.py`
2. **实现 Finding 抽取**：`python eval/extract_findings.py`（待实现）
3. **运行端到端评测**：`python eval/eval_end_to_end.py`（待实现）

## 技术细节

### 为什么需要 canonical_name？

统一事件标识，用于 Finding 匹配。例如：
- Agent 输出："OpenAI 发布了 GPT5.4"
- 通过 aliases 匹配到 canonical_name = "GPT-5.4"
- 判断为命中 ground truth 事件

### 为什么需要 aliases？

支持模糊匹配，提高召回率。例如：
- Ground truth: "GPT-5.4"
- Agent 可能输出: "GPT5.4", "gpt-5.4", "OpenAI GPT-5.4"
- 通过 aliases 都能匹配成功

### 为什么需要 gold_evidence？

快速验证 Finding 的证据支撑，避免重复读取完整文章：
- 完整文章：1000-2000 tokens
- Gold evidence：200-300 tokens
- 节省 80-90% 的 LLM 成本

### 为什么需要反向索引？

提高查询效率：
- 没有索引：O(事件数 × 平均文章数) ≈ O(1000)
- 有索引：O(1)
- 在 Finding 匹配时需要频繁查询
