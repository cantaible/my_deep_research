# OpenSearch 接入说明

## 目标

用 OpenSearch 替换当前手写 BM25，作为 RAG 的词法检索后端：

- 中文优先：使用内置 `cjk` analyzer
- 英文补充：使用 `english` analyzer
- 多字段加权：`title > summary > preview > body`

## 启动服务

```bash
docker compose -f docker-compose.opensearch.yml up -d
```

健康检查：

```bash
curl http://127.0.0.1:9200
```

## 构建词法索引

```bash
python rag/scripts/build_opensearch_index.py --force
```

## 运行方式

默认词法后端已切到 `opensearch`。

可选环境变量：

```bash
export LEXICAL_BACKEND=opensearch
export OPENSEARCH_URL=http://127.0.0.1:9200
export OPENSEARCH_INDEX_NAME=news_articles_v1
```

如果要临时回退旧实现：

```bash
export LEXICAL_BACKEND=bm25
```

## 当前实现边界

- 仍保留 Chroma 向量检索与本地 reranker
- OpenSearch 仅接管词法检索分支
- 中文 analyzer 先采用内置 `cjk`，未引入额外中文插件
