"""Phase 1 验证测试：确保所有基础设施模块正确加载。"""

import sys
sys.path.insert(0, "src")

def test_state():
    """验证 state.py 所有类 + reducer 逻辑。"""
    from state import (ConductResearch, ResearchComplete, Summary,
        ClarifyWithUser, ResearchQuestion, AgentInputState, AgentState,
        SupervisorState, ResearcherState, ResearcherOutputState,
        override_reducer)
    # 测试 override_reducer 两种模式
    assert override_reducer([1], {"type": "override", "value": [9]}) == [9]
    assert override_reducer([1], [2]) == [1, 2]
    print("✅ state.py: 5 Schema + 4 State + reducer 正确")

def test_config():
    """验证 configuration.py 默认值与原版一致。"""
    from configuration import Configuration, SearchAPI, MCPConfig
    cfg = Configuration()
    assert cfg.search_api == SearchAPI.TAVILY
    assert cfg.research_model == "openai:gpt-4.1"
    assert cfg.max_react_tool_calls == 10
    assert cfg.max_researcher_iterations == 6
    assert cfg.max_concurrent_research_units == 5
    print("✅ configuration.py: 默认值全部正确")

def test_prompts():
    """验证 prompts.py 7 个模板可导入且包含占位符。"""
    from prompts import (clarify_with_user_instructions,
        transform_messages_into_research_topic_prompt,
        lead_researcher_prompt, research_system_prompt,
        compress_research_system_prompt,
        compress_research_simple_human_message,
        final_report_generation_prompt, summarize_webpage_prompt)
    assert "{date}" in clarify_with_user_instructions
    assert "{research_brief}" in final_report_generation_prompt
    assert "{mcp_prompt}" in research_system_prompt
    print("✅ prompts.py: 7 个模板加载成功")

def test_utils():
    """验证 utils.py 所有公开函数可导入。"""
    from utils import (think_tool, get_today_str, get_config_value,
        get_api_key_for_model, get_notes_from_tool_calls,
        tavily_search, get_all_tools, is_token_limit_exceeded,
        get_model_token_limit, remove_up_to_last_ai_message,
        openai_websearch_called, anthropic_websearch_called)
    assert len(get_today_str()) > 0
    assert get_model_token_limit("openai:gpt-4o") == 128000
    print("✅ utils.py: 12 个函数加载成功")

if __name__ == "__main__":
    test_state()
    test_config()
    test_prompts()
    test_utils()
    print("\n🎉 Phase 1 全部验证通过！")
