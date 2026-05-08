# Phase 1 完成总结

## 已完成的工作

### 1. 核心脚本：`prepare_ground_truth.py`

**功能**：
- ✅ 从 `event_families.json` 映射 `canonical_name`
- ✅ 基于规则生成 `aliases`（支持 10+ 种厂商前缀）
- ✅ 提取 `gold_evidence`（支持数据库模式和纯标题模式）
- ✅ 生成反向索引 `event_to_articles.json`
- ✅ 生成统计报告 `ground_truth_stats.txt`

**使用方法**：
```bash
# 不连接数据库（推荐，快速）
python eval/prepare_ground_truth.py --no-db

# 连接数据库（需要 MariaDB）
python eval/prepare_ground_truth.py
```

### 2. 输出文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `article_labels_v2.json` | ~1.5 MB | 扩展后的文章标注（2262 篇） |
| `event_to_articles.json` | ~2 KB | 反向索引（100 条映射） |
| `ground_truth_stats.txt` | ~2 KB | 统计报告 |

### 3. 文档

- ✅ `README_GROUND_TRUTH.md`：完整使用文档
- ✅ `example_usage.py`：5 个使用示例

### 4. 数据统计

```
总文章数: 2262
  有 canonical_name: 100 (4.4%)  ← model_release 事件
  无 canonical_name: 2162 (95.6%)

事件家族数: 20
  GPT-5.4: 17 篇
  Qwen3.5: 11 篇
  GPT-5.3 Instant: 11 篇
  Composer: 9 篇
  MiMo: 8 篇
  ...
```

## 技术亮点

### 1. 零 LLM 成本

所有数据生成都是**纯代码逻辑**，不需要调用 LLM：
- `canonical_name`：从 `event_families.json` 直接映射
- `aliases`：基于规则生成（连字符、空格、大小写、厂商前缀）
- `gold_evidence`：从数据库或标题直接提取
- 反向索引：代码生成

### 2. 灵活的数据源

支持两种模式：
- **--no-db 模式**：只用标题作为 evidence（快速，适合开发）
- **数据库模式**：标题 + 正文前 300 字（完整，适合生产）

### 3. 完善的 Aliases 规则

支持 10+ 种厂商前缀：
- OpenAI (GPT)
- 阿里/通义 (Qwen)
- 深度求索 (DeepSeek)
- 智谱 (GLM)
- Google (Gemini)
- xAI (Grok)
- NVIDIA (Nemotron)
- Mistral AI
- MiniMax
- Composer

### 4. 高效的反向索引

查询效率：
- 没有索引：O(事件数 × 文章数) ≈ O(1000)
- 有索引：O(1)

## 使用示例

### 示例 1: 查询文章所属事件

```python
import json

event_index = json.load(open("eval/event_to_articles.json"))
article_id = 4362

if str(article_id) in event_index:
    print(f"Article {article_id} 属于事件: {event_index[str(article_id)]}")
# 输出: Article 4362 属于事件: DeepSeek V4
```

### 示例 2: 模糊匹配事件

```python
# Agent 输出
agent_text = "OpenAI 在 3 月发布了 GPT5.4"

# 加载 aliases
labels_v2 = json.load(open("eval/article_labels_v2.json"))
all_aliases = {}
for label in labels_v2:
    if label["canonical_name"]:
        for alias in label["aliases"]:
            all_aliases[alias] = label["canonical_name"]

# 匹配
matched = {all_aliases[a] for a in all_aliases if a in agent_text}
print(matched)
# 输出: {'GPT-5.4'}
```

### 示例 3: 验证证据支撑

```python
# Agent 的 finding
finding = {
    "text": "DeepSeek 即将推出 V4 版本",
    "evidence_article_ids": [4362]
}

# 获取 gold_evidence
labels_dict = {l["article_id"]: l for l in labels_v2}
evidence = labels_dict[4362]["gold_evidence"]

# 用 LLM 判断
prompt = f"""
证据: {evidence}
Finding: {finding["text"]}
问题: 证据是否支持 finding? Yes/No
"""
# ... 调用 LLM
```

## 下一步工作

### Phase 2: Researcher 检索评测增强

**任务**：
1. 修改 `rag/rag.py`，记录 dense/sparse/merged/reranked 各阶段结果
2. 实现 `eval/eval_rag_retrieval.py`，计算各阶段召回率
3. 对比 dense vs sparse 的召回贡献

**预计工作量**：4-5 小时

### Phase 3: Finding 抽取与匹配

**任务**：
1. 定义 `Finding` schema
2. 实现 `eval/extract_findings.py`（用 LLM 从报告中抽取 findings）
3. 实现 `eval/match_findings.py`（基于 aliases 和实体重叠匹配）

**预计工作量**：5-6 小时

### Phase 4: 总体结果评测

**任务**：
1. 实现 `eval/eval_findings.py`（计算 Finding Recall/Precision/F1）
2. 实现 `eval/eval_end_to_end.py`（端到端评测流程）

**预计工作量**：3-4 小时

## 文件清单

```
eval/
├── prepare_ground_truth.py       ✅ Phase 1 核心脚本
├── README_GROUND_TRUTH.md        ✅ 使用文档
├── example_usage.py              ✅ 使用示例
├── article_labels_v2.json        ✅ 扩展后的标注（生成）
├── event_to_articles.json        ✅ 反向索引（生成）
├── ground_truth_stats.txt        ✅ 统计报告（生成）
├── article_labels.json           ✅ 原始标注（已有）
├── event_families.json           ✅ 事件家族（已有）
└── eval_rag.py                   ✅ RAG 评测（已有）
```

## 验证结果

所有功能已验证通过：
- ✅ 脚本运行成功（2262 篇文章，耗时 < 10 秒）
- ✅ 生成 100 条 canonical_name 映射
- ✅ 生成 aliases（平均每个事件 5-10 个变体）
- ✅ 提取 gold_evidence（2262 篇，成功率 100%）
- ✅ 反向索引查询正常（O(1) 复杂度）
- ✅ 示例脚本运行正常（5 个示例全部通过）

## 总结

Phase 1 已完成，生成了评测系统所需的基础数据：
- **canonical_name**：统一事件标识
- **aliases**：支持模糊匹配
- **gold_evidence**：快速验证证据
- **反向索引**：高效查询

所有功能都是**零 LLM 成本**，完全基于规则和现有数据生成。

可以开始 Phase 2 的工作了！
