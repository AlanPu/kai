"""
阶段 2 测试：文本能力（planner + corrector）。

分两部分：
  · 纯逻辑测试 —— 不调网络，快速、稳定，CI 必跑
  · 联网测试   —— 真实调用模型，用 -m live 选择性运行
"""

import pytest

from app.core.content import Content, classify, extract_article
from app.core.llm import LLMError, parse_json_loose
from app.services.corrector import Corrector
from app.services.planner import (Plan, Planner, Topic, build_tutor_instructions,
                                  is_open_question, validate_plan)


# ============================================================
#  输入判定
# ============================================================

@pytest.mark.parametrize("text,expected", [
    ("聊一聊人工智能", "topic"),
    ("I went to the park.", "topic"),
    ("https://example.com/a", "url"),
    ("example.com/post", "url"),
    ("AI changes things. It affects jobs.", "passage"),
])
def test_classify(text, expected):
    assert classify(text) == expected


def test_classify_long_text_is_article():
    long_text = ("Machine learning has transformed industries. " * 40)
    assert classify(long_text) == "article"


# ============================================================
#  HTML 提取
# ============================================================

def test_extract_strips_scripts_and_chrome():
    html = ("<html><head><title>T</title>"
            "<script>bad()</script><style>.x{}</style></head>"
            "<body><nav>menu</nav><p>Real content here.</p>"
            "<footer>copyright</footer></body></html>")
    title, body = extract_article(html)
    assert title == "T"
    assert "Real content here." in body
    for junk in ["bad()", ".x{}", "menu", "copyright"]:
        assert junk not in body, f"未清除: {junk}"


def test_extract_handles_empty():
    assert extract_article("") == (None, "")


# ============================================================
#  JSON 容错
# ============================================================

@pytest.mark.parametrize("raw,expected", [
    ('{"a":1}', {"a": 1}),
    ('```json\n{"b":2}\n```', {"b": 2}),
    ('结果如下：{"c":3} 完', {"c": 3}),
    ('[1,2,3]', [1, 2, 3]),
])
def test_parse_json_loose(raw, expected):
    assert parse_json_loose(raw) == expected


def test_parse_json_loose_raises_on_garbage():
    with pytest.raises(LLMError):
        parse_json_loose("这不是 JSON")


# ============================================================
#  开放式问题判定（防误判）
# ============================================================

@pytest.mark.parametrize("q,expected", [
    ("What do you think?", True),
    ("Tell me about your day.", True),
    ("How do you relax?", True),
    # 以 Have 开头但有追问 → 开放式（曾经的误判案例）
    ("Have you ever felt tension? What happened?", True),
    ("Do you like coffee?", False),
    ("Did you go?", False),
    ("Are you a student?", False),
    ("Can you swim?", False),
])
def test_is_open_question(q, expected):
    assert is_open_question(q) is expected


def test_validate_plan_reports_problems():
    p = Plan(opening="", topics=[Topic(title="t", prompt="Do you like it?")])
    issues = validate_plan(p)
    assert any("开场白" in i for i in issues)
    assert any("封闭式" in i for i in issues)


def test_validate_plan_clean():
    p = Plan(opening="Hi!", topics=[
        Topic(title=f"t{i}", prompt=f"What about topic {i}?") for i in range(4)])
    assert validate_plan(p) == []


# ============================================================
#  计划序列化
# ============================================================

def test_plan_roundtrip():
    p = Plan(opening="Hi", background="bg", vocabulary=["a", "b"],
             topics=[Topic(title="t", prompt="What?", why="w")])
    p2 = Plan.from_dict(p.to_dict())
    assert p2.opening == p.opening
    assert p2.vocabulary == p.vocabulary
    assert p2.topics[0].prompt == "What?"


def test_plan_from_dict_tolerates_junk():
    """模型偶尔多给字段，不应崩溃。"""
    p = Plan.from_dict({
        "opening": "hi",
        "topics": [{"title": "t", "prompt": "What?", "unknown_field": 1}],
    })
    assert p.topics[0].prompt == "What?"


# ============================================================
#  纠错结果归一化（不调网络）
# ============================================================

def test_corrector_normalize_basic():
    items = Corrector._normalize({"corrections": [
        {"kind": "grammar", "severity": "critical",
         "original": "I go", "suggestion": "I went"},
    ]})
    assert len(items) == 1
    assert items[0].kind == "grammar"
    assert items[0].severity == "critical"


def test_corrector_strips_word_from_non_pronunciation():
    """word 字段只属于发音类，其他类型要清掉，避免污染统计。"""
    items = Corrector._normalize({"corrections": [
        {"kind": "grammar", "severity": "minor", "original": "a",
         "suggestion": "b", "word": "should-be-removed"},
    ]})
    assert items[0].word is None


def test_corrector_pronunciation_without_word_downgraded():
    """发音问题但没给具体单词 → 无法定位，降级为 ignore。"""
    items = Corrector._normalize({"corrections": [
        {"kind": "pronunciation", "severity": "critical",
         "original": "x", "suggestion": "y"},
    ]})
    assert items[0].severity == "ignore"


def test_corrector_rejects_invalid_enums():
    items = Corrector._normalize({"corrections": [
        {"kind": "not_a_kind", "severity": "not_a_sev",
         "original": "a", "suggestion": "b"},
    ]})
    assert items[0].kind == "grammar"
    assert items[0].severity == "minor"


def test_corrector_skips_incomplete_items():
    items = Corrector._normalize({"corrections": [
        {"kind": "grammar", "original": "a"},           # 缺 suggestion
        {"kind": "grammar", "suggestion": "b"},         # 缺 original
        {"kind": "grammar", "original": "c", "suggestion": "d"},
    ]})
    assert len(items) == 1


def test_corrector_handles_bare_list():
    items = Corrector._normalize([
        {"kind": "vocabulary", "original": "a", "suggestion": "b"}])
    assert len(items) == 1


# ============================================================
#  系统提示词体现需求
# ============================================================

def test_instructions_encode_requirements():
    """提示词必须包含需求 2/3/4/5 的关键约束。"""
    plan = Plan(opening="hi", background="bg",
                topics=[Topic(title="t", prompt="What?")],
                vocabulary=["word"])
    ins = build_tutor_instructions(plan, minutes=30)

    # 需求 2：多给我说的机会
    assert "SHORT" in ins
    assert "60%" in ins
    # 需求 3：纠错但不打断
    assert "Do NOT interrupt" in ins
    # 需求 4：发音只纠明显错误
    assert "CLEAR word-level" in ins
    assert "accents" in ins
    # 话题注入
    assert "What?" in ins


def test_instructions_include_profile_when_given():
    plan = Plan(opening="hi", topics=[Topic(title="t", prompt="Q?")])
    ins = build_tutor_instructions(plan, profile_summary="他喜欢跑步")
    assert "他喜欢跑步" in ins


def test_instructions_omit_profile_when_empty():
    plan = Plan(opening="hi", topics=[Topic(title="t", prompt="Q?")])
    ins = build_tutor_instructions(plan, profile_summary="")
    assert "already know" not in ins


# ============================================================
#  联网测试（pytest -m live）
# ============================================================

@pytest.mark.live
def test_live_planner_produces_open_topics():
    content = Content(
        kind="passage", raw="x",
        text="Remote work became common after 2020. Companies allow employees "
             "to work from home. Managers worry about collaboration.")
    plan = Planner().plan(content, n_topics=4)
    assert len(plan.topics) >= 3
    assert plan.opening

    # 这里刻意不断言 validate_plan() == []。
    # validate_plan 里的 is_open_question 是启发式判断，
    # 而模型输出是自由的 —— 偶尔会有一个措辞被判成"封闭式"，
    # 但那个话题实际完全可用。断言为零容忍会让测试随机变红，
    # 而真正要保证的是"话题能让人开口"，所以只要求绝大多数合格。
    problems = validate_plan(plan)
    open_issues = [p for p in problems if "封闭式" in p]
    assert len(open_issues) == 0 or len(open_issues) <= max(1, len(plan.topics) // 4), \
        f"过多话题疑似封闭: {open_issues}"
    assert not [p for p in problems if "缺少" in p], f"有缺失字段: {problems}"


@pytest.mark.live
def test_live_corrector_catches_clear_grammar_error():
    items = Corrector().check("I go to park yesterday.")
    assert items, "明显语法错误未被检出"
    assert any(i.kind == "grammar" for i in items)
    assert any(i.severity == "critical" for i in items)


@pytest.mark.live
def test_live_corrector_silent_on_correct_sentence():
    assert Corrector().check("I think this is a good idea.") == []


@pytest.mark.live
def test_live_corrector_pronunciation_only_when_obvious():
    """需求 4：正常句子不得报发音错误。"""
    for text in ["I want to drink some water.",
                 "The weather is really nice today."]:
        items = Corrector().check(text)
        pron = [i for i in items if i.kind == "pronunciation"]
        assert not pron, f"不该报发音错误: {text} → {pron}"
