"""分析 AutoResearcher 运行日志，提供优化建议。"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def parse_timestamp(ts_str):
    """解析时间戳"""
    try:
        return datetime.fromisoformat(ts_str)
    except:
        return None


def analyze_events(events_file: Path):
    """分析 events.jsonl 文件"""

    # 统计数据
    node_calls = Counter()
    node_start_times = {}
    node_durations = defaultdict(list)
    llm_calls = Counter()
    rag_queries = []
    web_searches = []

    # 读取事件
    with open(events_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            event = json.loads(line)
            event_type = event.get('type', '')
            node = event.get('node', '')
            ts = event.get('ts', '')

            # 统计节点调用
            if event_type == 'node_start':
                node_calls[node] += 1
                # 记录开始时间（用于计算持续时间）
                key = f"{node}_{ts}"
                node_start_times[key] = parse_timestamp(ts)

            elif event_type == 'node_end':
                # 计算持续时间
                key = f"{node}_{ts}"
                # 简化处理：找最近的 start 时间
                for start_key, start_time in list(node_start_times.items()):
                    if start_key.startswith(node + "_"):
                        end_time = parse_timestamp(ts)
                        if start_time and end_time:
                            duration = (end_time - start_time).total_seconds()
                            node_durations[node].append(duration)
                        del node_start_times[start_key]
                        break

            # 统计 LLM 调用
            elif event_type == 'llm_start':
                model = event.get('model', 'unknown')
                llm_calls[model] += 1

            # 提取 RAG 查询（从 llm_stream 中推断）
            elif event_type == 'llm_stream' and node == 'rag_search':
                token = event.get('token', '')
                if 'query' in token.lower():
                    # 简化处理：标记有 RAG 查询
                    pass

    return {
        'node_calls': node_calls,
        'node_durations': node_durations,
        'llm_calls': llm_calls,
        'rag_queries': rag_queries,
        'web_searches': web_searches,
    }


def print_analysis(stats: dict, report_file: Path, run_meta_file: Path):
    """打印分析结果和优化建议"""

    print("=" * 80)
    print("AutoResearcher 运行分析")
    print("=" * 80)

    # 读取运行元数据
    if run_meta_file.exists():
        meta = json.loads(run_meta_file.read_text(encoding='utf-8'))
        print(f"\n运行信息:")
        print(f"  主题: {meta.get('topic', 'N/A')[:80]}...")
        print(f"  耗时: {meta.get('elapsed_seconds', 0):.1f} 秒 ({meta.get('elapsed_seconds', 0) / 60:.1f} 分钟)")
        print(f"  报告长度: {meta.get('report_length', 0)} 字符")
        print(f"  是否完成: {'是' if meta.get('completed') else '否'}")

    print(f"\n节点调用统计（Top 15）:")
    for node, count in stats['node_calls'].most_common(15):
        avg_duration = ""
        if node in stats['node_durations'] and stats['node_durations'][node]:
            avg = sum(stats['node_durations'][node]) / len(stats['node_durations'][node])
            avg_duration = f"(平均 {avg:.2f}s)"
        print(f"  {node:30s} {count:5d} 次 {avg_duration}")

    print(f"\nLLM 调用统计:")
    total_llm = sum(stats['llm_calls'].values())
    for model, count in stats['llm_calls'].most_common():
        print(f"  {model:30s} {count:5d} 次")
    print(f"  总计: {total_llm} 次")

    # 读取报告
    if report_file.exists():
        report = report_file.read_text(encoding='utf-8')
        print(f"\n报告统计:")
        print(f"  字符数: {len(report)}")
        print(f"  行数: {len(report.splitlines())}")
        print(f"  章节数: {report.count('##')}")

    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)

    suggestions = []

    # 1. 运行时间分析
    if run_meta_file.exists():
        meta = json.loads(run_meta_file.read_text(encoding='utf-8'))
        elapsed = meta.get('elapsed_seconds', 0)
        if elapsed > 600:  # 超过 10 分钟
            suggestions.append(
                f"⚠️  运行时间较长 ({elapsed / 60:.1f} 分钟)\n"
                "   建议:\n"
                "   - 检查是否有过多的重试或循环\n"
                "   - 考虑并行化独立的研究任务\n"
                "   - 优化 LLM 调用次数和 prompt 长度"
            )

    # 2. LLM 调用效率
    total_llm = sum(stats['llm_calls'].values())
    if total_llm > 50:
        suggestions.append(
            f"⚠️  LLM 调用次数较多 ({total_llm} 次)\n"
            "   建议:\n"
            "   - 检查是否有重复的 LLM 调用\n"
            "   - 考虑缓存中间结果\n"
            "   - 合并相似的查询或推理任务"
        )

    # 3. 节点调用模式分析
    clarify_count = stats['node_calls'].get('clarify_with_user', 0)
    plan_count = stats['node_calls'].get('plan', 0)
    research_count = stats['node_calls'].get('research', 0)

    if clarify_count > 3:
        suggestions.append(
            f"⚠️  用户澄清次数较多 ({clarify_count} 次)\n"
            "   建议:\n"
            "   - 优化初始 prompt，提供更明确的指令\n"
            "   - 在 clarify 节点中一次性收集所有必要信息"
        )

    if plan_count > 5:
        suggestions.append(
            f"⚠️  规划节点调用次数较多 ({plan_count} 次)\n"
            "   建议:\n"
            "   - 检查是否有过多的重新规划\n"
            "   - 优化规划策略，减少不必要的调整"
        )

    # 4. 报告质量分析
    if report_file.exists():
        report = report_file.read_text(encoding='utf-8')
        if len(report) < 3000:
            suggestions.append(
                f"⚠️  报告内容较少 ({len(report)} 字符)\n"
                "   建议:\n"
                "   - 检查是否有足够的信息源\n"
                "   - 优化信息抽取和综合策略"
            )
        elif len(report) > 20000:
            suggestions.append(
                f"⚠️  报告内容较多 ({len(report)} 字符)\n"
                "   建议:\n"
                "   - 考虑更激进的信息压缩\n"
                "   - 优化报告结构，突出重点"
            )

    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion}")
    else:
        print("\n✅ 运行效率良好，暂无明显优化点")

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)

    # 计算一些关键指标
    if run_meta_file.exists():
        meta = json.loads(run_meta_file.read_text(encoding='utf-8'))
        elapsed = meta.get('elapsed_seconds', 0)
        report_length = meta.get('report_length', 0)

        if elapsed > 0:
            chars_per_second = report_length / elapsed
            print(f"\n生成效率: {chars_per_second:.1f} 字符/秒")

        if total_llm > 0:
            chars_per_llm = report_length / total_llm
            print(f"LLM 效率: {chars_per_llm:.1f} 字符/LLM调用")

    print()


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/analyze_run.py <run_dir>")
        print("示例: python scripts/analyze_run.py 'logs/给我调研一下2026年3月1日到3月31日都有哪些大模型发布-20260411-175457'")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    events_file = run_dir / "events.jsonl"
    report_file = run_dir / "report.md"
    run_meta_file = run_dir / "run_meta.json"

    if not events_file.exists():
        print(f"❌ 找不到 {events_file}")
        sys.exit(1)

    print(f"分析运行目录: {run_dir.name}")

    stats = analyze_events(events_file)
    print_analysis(stats, report_file, run_meta_file)


if __name__ == "__main__":
    main()
