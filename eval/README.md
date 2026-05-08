# Eval 目录说明

评测系统的所有脚本、数据和文档。

---

## 📋 核心脚本

| 文件 | 用途 |
|------|------|
| `prepare_ground_truth.py` | **生成扩展后的 ground truth 数据**<br>添加 canonical_name、aliases、gold_evidence 字段<br>用法：`python eval/prepare_ground_truth.py --no-db` |
| `eval_rag.py` | **RAG 子图评测**<br>运行 RAG 子图并计算事件级别的 Recall/Precision/F1/NDCG<br>用法：`python eval/eval_rag.py` |
| `eval_rag_retrieval.py` | **RAG 检索细粒度评测**（✨ Phase 2 新增）<br>分析 dense/sparse/merged/reranked 各阶段的召回表现<br>用法：`python eval/eval_rag_retrieval.py` |
| `eval_findings.py` | **Finding 级别评测**（✨ Phase 3 新增）<br>从研究报告中抽取 findings 并与 ground truth 匹配<br>用法：`python eval/eval_findings.py` |
| `finding_schema.py` | **Finding 数据模型**（✨ Phase 3 新增）<br>定义 Finding 和 FindingMatch 的数据结构 |
| `finding_extractor.py` | **Finding 抽取器**（✨ Phase 3 新增）<br>使用 LLM 从研究报告中抽取结构化事件信息 |
| `finding_matcher.py` | **Finding 匹配器**（✨ Phase 3 新增）<br>基于 aliases 和实体重叠匹配 ground truth |
| `label_articles.py` | **文章标注工具**<br>用 LLM 对文章进行事件类型标注（已完成，生成了 article_labels.json） |
| `example_usage.py` | **使用示例**<br>演示如何使用扩展后的 ground truth 数据<br>用法：`python eval/example_usage.py` |

---

## 📊 数据文件

### Ground Truth 数据

| 文件 | 说明 |
|------|------|
| `article_labels.json` | **原始文章标注**（702 KB）<br>2262 篇文章的事件类型、实体标注 |
| `article_labels_v2.json` | **扩展后的文章标注**（1.1 MB）<br>在原始标注基础上添加了 canonical_name、aliases、gold_evidence |
| `event_families.json` | **事件家族定义**（3 KB）<br>20 个模型发布事件，每个事件包含多篇报道文章 |
| `event_to_articles.json` | **反向索引**（2 KB）<br>article_id → canonical_name 的快速查询索引 |

### 中间数据

| 文件 | 说明 |
|------|------|
| `reviewed_events.json` | 人工审核的事件列表 |
| `model_release_review.csv` | 模型发布事件的审核表格 |

---

## 📈 统计报告

| 文件 | 说明 |
|------|------|
| `ground_truth_stats.txt` | Ground truth 数据统计报告<br>包含文章数、事件数、事件类型分布等 |

---

## 📖 文档

| 文件 | 说明 |
|------|------|
| `README.md` | **本文件**<br>eval 目录的文件说明 |
| `README_GROUND_TRUTH.md` | **Ground Truth 数据准备文档**<br>详细说明如何使用 prepare_ground_truth.py |
| `PHASE1_SUMMARY.md` | **Phase 1 完成总结**<br>数据准备阶段的工作总结 |
| `PHASE2_PLAN.md` | **Phase 2 实施计划**<br>Researcher 检索评测增强的详细计划 |
| `PHASE2_DATA_FLOW.md` | **Phase 2 数据流说明**<br>检索详情的保存位置和读取方式 |
| `PHASE2_SUMMARY.md` | **Phase 2 完成总结**（✨ 新增）<br>细粒度检索评测的工作总结 |
| `PHASE3_SUMMARY.md` | **Phase 3 完成总结**（✨ 新增）<br>Finding 抽取与匹配评测的工作总结 |
| `EXPERIMENTS.md` | **实验记录**<br>RAG 分块策略对比实验 |

---

## 🌐 可视化文件

| 文件 | 说明 |
|------|------|
| `event_review.html` | 事件审核可视化页面 |
| `family_merge.html` | 事件家族合并可视化页面 |

---

## 快速开始

### 1. 生成扩展后的 ground truth 数据

```bash
python eval/prepare_ground_truth.py --no-db
```

**输出**：
- `article_labels_v2.json`
- `event_to_articles.json`
- `ground_truth_stats.txt`

### 2. 查看使用示例

```bash
python eval/example_usage.py
```

### 3. 运行 RAG 评测

```bash
python eval/eval_rag.py
```

### 4. 运行细粒度检索评测（✨ Phase 2 新增）

```bash
python eval/eval_rag_retrieval.py
```

**输出**：
- `logs/{run_dir}/retrieval_details.json`（检索详情）
- `logs/{run_dir}/RETRIEVAL_REPORT.md`（评测报告）
- `logs/{run_dir}/retrieval_metrics.json`（评测指标）

### 5. 运行 Finding 级别评测（✨ Phase 3 新增）

```bash
python eval/eval_findings.py
```

**输出**：
- `logs/{run_dir}/FINDING_REPORT.md`（评测报告）
- `logs/{run_dir}/finding_metrics.json`（评测指标）
- `logs/{run_dir}/finding_matches.json`（匹配详情）

---

## 数据流

```
原始数据
├── article_labels.json (原始标注)
└── event_families.json (事件家族)
         │
         ▼ prepare_ground_truth.py
扩展数据
├── article_labels_v2.json (扩展标注)
├── event_to_articles.json (反向索引)
└── ground_truth_stats.txt (统计报告)
         │
         ▼ eval_rag.py
RAG 运行结果
├── compressed.md (压缩报告)
├── raw_results.json (原始结果)
└── retrieval_details.json (检索详情)
         │
         ├─→ eval_rag_retrieval.py (Phase 2)
         │   ├── RETRIEVAL_REPORT.md
         │   └── retrieval_metrics.json
         │
         └─→ eval_findings.py (Phase 3)
             ├── FINDING_REPORT.md
             ├── finding_metrics.json
             └── finding_matches.json
```

---

## 目录结构

```
eval/
├── README.md                      ← 本文件
├── README_GROUND_TRUTH.md         ← Ground Truth 文档
├── PHASE1_SUMMARY.md              ← Phase 1 总结
├── PHASE2_SUMMARY.md              ← Phase 2 总结
├── PHASE3_SUMMARY.md              ← Phase 3 总结（✨ 新增）
├── EXPERIMENTS.md                 ← 实验记录
│
├── prepare_ground_truth.py        ← 数据准备脚本
├── eval_rag.py                    ← RAG 评测脚本
├── eval_rag_retrieval.py          ← RAG 检索细粒度评测（Phase 2）
├── eval_findings.py               ← Finding 级别评测（Phase 3）（✨ 新增）
├── finding_schema.py              ← Finding 数据模型（Phase 3）（✨ 新增）
├── finding_extractor.py           ← Finding 抽取器（Phase 3）（✨ 新增）
├── finding_matcher.py             ← Finding 匹配器（Phase 3）（✨ 新增）
├── label_articles.py              ← 文章标注工具
├── example_usage.py               ← 使用示例
│
├── article_labels.json            ← 原始标注
├── article_labels_v2.json         ← 扩展标注
├── event_families.json            ← 事件家族
├── event_to_articles.json         ← 反向索引
├── ground_truth_stats.txt         ← 统计报告
│
├── reviewed_events.json           ← 审核数据
├── model_release_review.csv       ← 审核表格
├── event_review.html              ← 可视化页面
└── family_merge.html              ← 可视化页面
```

---

## 下一步

- ✅ **Phase 1**：Ground Truth 数据准备（已完成）
- ✅ **Phase 2**：Researcher 检索评测增强（已完成）
- ✅ **Phase 3**：Finding 抽取与匹配（已完成）
- ⏸️ **Phase 4**：总体结果评测（可选，整合 Phase 2 和 Phase 3）
