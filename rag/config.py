"""
RAG 系统配置文件。

集中管理所有配置项：数据库连接、向量存储路径、模型选择、检索参数。
修改配置只需改这一个文件，其他模块通过 import config 引用。
"""
import os
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
# RAG_DIR: 本文件所在的 rag/ 目录，作为其他路径的基准
RAG_DIR = Path(__file__).parent
# VECTORDB_DIR: ChromaDB 向量数据库的持久化存储目录
# ChromaDB 会在这个目录下保存索引文件，删掉就需要重新 build
VECTORDB_DIR = RAG_DIR / "vectordb"

# ============================================================
# MariaDB 连接配置（数据源）
# ============================================================
# 这是从生产环境拉下来的只读副本，运行在 Docker 容器中
# 容器名: news-reader-local-db，映射到本机 3307 端口
DB_HOST = "127.0.0.1"
DB_PORT = 3307
DB_USER = "root"
DB_PASSWORD = "rootpass"
DB_NAME = "news_reader"  # 数据库名，包含 news_article 等表

# ============================================================
# ChromaDB 向量数据库配置
# ============================================================
# Collection 相当于表，所有文章的向量都存在这一个 collection 里
COLLECTION_NAME = "news_articles"

# ============================================================
# Embedding 模型配置
# ============================================================
# bge-m3: 多语言 embedding 模型，中英文效果都好
# 首次运行会自动从 HuggingFace 下载 (~2.3GB)，之后用本地缓存
EMBEDDING_MODEL = "BAAI/bge-m3"

# ============================================================
# Reranker 模型配置
# ============================================================
# bge-reranker-v2-m3: 本地部署成本较低、支持中英文的 reranker
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LENGTH = 512

# ============================================================
# 分块 (Chunking) 配置 (用于方案 A 深度检索)
# ============================================================
CHUNKS_COLLECTION_NAME = "news_chunks"
CHUNK_SIZE = 1000        # 每个 Chunk 的最大长度
CHUNK_OVERLAP = 200      # 块与块之间的重叠字数，避免关键语境被切断

# ============================================================
# 检索默认参数
# ============================================================
# 每次检索返回的最大结果数
DEFAULT_MAX_RESULTS = 10
# 第一阶段召回的候选池大小倍率。最终输出前会再经过 reranker。
RETRIEVAL_CANDIDATE_MULTIPLIER = 4
# 默认时间范围（天），比如 30 = 只搜最近一个月的文章
DEFAULT_TIME_RANGE_DAYS = 30
# 返回结果中 raw_content 预览的最大字符数
RAW_CONTENT_PREVIEW_LENGTH = 500

# ============================================================
# 词法检索后端配置
# ============================================================
# 可选值：
# - opensearch: 强制使用 OpenSearch 作为词法检索后端
# - auto: 优先 OpenSearch，不可用时回退到本地 BM25
# - bm25: 仅使用本地 BM25（兼容旧实现）
LEXICAL_BACKEND = os.getenv("LEXICAL_BACKEND", "opensearch").strip().lower()

# ============================================================
# OpenSearch 配置
# ============================================================
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://127.0.0.1:9200").strip()
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "news_articles_v1").strip()
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME", "").strip()
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "").strip()
OPENSEARCH_TIMEOUT = int(os.getenv("OPENSEARCH_TIMEOUT", "30"))
