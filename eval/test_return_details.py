"""测试 rag_search 的 return_details 功能"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from rag_search import rag_search

# 测试 1: 默认行为（返回字符串）
print("测试 1: 默认行为（return_details=False）")
result = rag_search.invoke({
    "query": "GPT-5.4",
    "start_date": "2026-03-01",
    "end_date": "2026-03-31",
    "category": "AI",
    "top_k": 5,
    "return_details": False,
})
print(f"返回类型: {type(result)}")
print(f"返回内容（前 200 字符）: {result[:200] if isinstance(result, str) else 'N/A'}")
print()

# 测试 2: 返回详情（返回字典）
print("测试 2: 返回详情（return_details=True）")
result = rag_search.invoke({
    "query": "GPT-5.4",
    "start_date": "2026-03-01",
    "end_date": "2026-03-31",
    "category": "AI",
    "top_k": 5,
    "return_details": True,
})
print(f"返回类型: {type(result)}")
if isinstance(result, dict):
    print(f"包含的键: {list(result.keys())}")
    print(f"formatted_output 长度: {len(result.get('formatted_output', ''))}")

    details = result.get("retrieval_details", {})
    print(f"\n检索详情:")
    print(f"  - query: {details.get('query')}")
    print(f"  - dense 召回: {details.get('dense', {}).get('count')} 篇")
    print(f"  - sparse 召回: {details.get('sparse', {}).get('count')} 篇")
    print(f"  - merged 候选池: {details.get('merged', {}).get('count')} 篇")
    print(f"  - reranked 最终: {details.get('reranked', {}).get('count')} 篇")

    # 显示前 5 个 article_ids
    print(f"\n  - dense article_ids (前5): {details.get('dense', {}).get('article_ids', [])[:5]}")
    print(f"  - sparse article_ids (前5): {details.get('sparse', {}).get('article_ids', [])[:5]}")
    print(f"  - reranked article_ids (前5): {details.get('reranked', {}).get('article_ids', [])[:5]}")
else:
    print("❌ 返回类型不是 dict")

print("\n✅ 测试完成")
