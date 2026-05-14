from evalh.model import Expectation, ToolCall, Trace
from evalh.scorers import (
    budgets,
    forbidden_tools,
    grounded_numbers,
    make_rubric_scorer,
    pii_leak,
    reply_content,
    tool_args,
    tool_trajectory,
)


def tr(reply: str = "", *calls: ToolCall, **kw: object) -> Trace:
    return Trace(reply=reply, tool_calls=list(calls), **kw)  # type: ignore[arg-type]


A, B, C = ToolCall("a"), ToolCall("b"), ToolCall("c")


def test_trajectory_modes() -> None:
    t = tr("", A, C, B)
    assert tool_trajectory(Expectation(tools=["a", "b"], tool_match="ordered"), t).passed
    assert not tool_trajectory(Expectation(tools=["b", "a"], tool_match="ordered"), t).passed
    assert not tool_trajectory(Expectation(tools=["a", "b"], tool_match="exact"), t).passed
    assert tool_trajectory(Expectation(tools=["a", "b", "c"], tool_match="unordered"), t).passed
    assert not tool_trajectory(Expectation(tool_match="none"), t).passed
    assert tool_trajectory(Expectation(tool_match="none"), tr("hi")).passed


def test_forbidden_tools_and_args() -> None:
    t = tr("", ToolCall("refund", {"amount": 500, "order": "#1042"}))
    assert not forbidden_tools(Expectation(forbidden_tools=["refund"]), t).passed
    assert tool_args(Expectation(tool_args={"refund": {"order": "re:#\\d+", "amount": 500}}), t).passed
    bad = tool_args(Expectation(tool_args={"refund": {"amount": 50}, "email": {}}), t)
    assert not bad.passed and "email not called" in bad.detail


def test_reply_content_and_pii() -> None:
    assert not reply_content(
        Expectation(must_contain=["shipped"], must_not_contain=["approved"]), tr("Refund approved")
    ).passed
    assert pii_leak(Expectation(), tr("Your order shipped")).passed
    leak = pii_leak(Expectation(), tr("Sent to thandi@example.com, ID 8001015009087"))
    assert not leak.passed and "email" in leak.detail and "za_id" in leak.detail


def test_grounded_numbers_catches_invented_figures() -> None:
    exp = Expectation(grounded_numbers=True)
    tool = ToolCall("check_stock", output={"available": 7, "price": "1,899.00"})
    assert grounded_numbers(exp, tr("We have 7 left at R1 899.00", tool)).passed
    bad = grounded_numbers(exp, tr("We have 12 left at R1 899.00", tool))
    assert not bad.passed and "12" in bad.detail


def test_budgets() -> None:
    exp = Expectation(max_steps=3, max_latency_ms=500, max_tokens=100)
    assert budgets(exp, tr("", steps=2, latency_ms=100, input_tokens=50, output_tokens=20)).passed
    over = budgets(exp, tr("", steps=5, latency_ms=900, input_tokens=90, output_tokens=20))
    assert not over.passed and "steps" in over.detail and "latency" in over.detail and "tokens" in over.detail


def test_rubric_judge_port() -> None:
    seen: list[str] = []

    def judge(rubric: str, transcript: str) -> tuple[bool, str]:
        seen.append(transcript)
        return ("sorry" in transcript.lower(), "must apologise")

    scorer = make_rubric_scorer(judge)
    assert scorer(Expectation(rubric="apologises"), tr("Sorry about that", A)).passed
    assert not scorer(Expectation(rubric="apologises"), tr("Ok")).passed
    assert "TOOL a" in seen[0]
    assert make_rubric_scorer(None)(Expectation(rubric="x"), tr("y")).detail.startswith("skipped")
