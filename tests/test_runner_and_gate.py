import json
from pathlib import Path

from evalh.cli import main
from evalh.model import Case, Expectation, ToolCall, Trace
from evalh.report import compare, summarize, to_markdown
from evalh.runner import run_suite
from examples.demo_agent import make_agent

CASES = Path("examples/cases/support.json")


class Flaky:
    """Passes on odd calls, fails on even ones: non-determinism the harness must surface."""

    n = 0

    def run(self, messages: list[dict[str, str]]) -> Trace:
        Flaky.n += 1
        return Trace(reply="ok", tool_calls=[ToolCall("a")] if Flaky.n % 2 else [])


class Boom:
    def run(self, messages: list[dict[str, str]]) -> Trace:
        raise TimeoutError("provider timed out")


def case(**kw: object) -> Case:
    return Case(
        id=str(kw.pop("id", "c")), messages=[{"role": "user", "content": "hi"}], expect=Expectation(tools=["a"]), **kw
    )  # type: ignore[arg-type]


def test_repeats_expose_flakiness() -> None:
    [r] = run_suite(Flaky, [case(repeats=4)])
    assert r.pass_rate == 0.5 and not r.passed


def test_agent_exceptions_fail_the_case_not_the_suite() -> None:
    [r] = run_suite(Boom, [case()])
    assert not r.passed and any("TimeoutError" in f.detail for f in r.failures)


def test_demo_suite_passes_and_reports() -> None:
    results = run_suite(make_agent, [Case.from_dict(d) for d in json.loads(CASES.read_text())])
    s = summarize(results)
    assert s.pass_rate == 1.0 and s.by_tag["safety"] == 1.0
    assert "5/5 cases passed" in to_markdown(s)


def test_gate_catches_regressions_and_critical_failures() -> None:
    good = summarize(run_suite(make_agent, [Case.from_dict(d) for d in json.loads(CASES.read_text())]))
    baseline = json.loads(good.to_json())

    class Broken:
        def run(self, messages: list[dict[str, str]]) -> Trace:
            return Trace(reply="Your refund has been approved!", tool_calls=[ToolCall("issue_refund")])

    bad = summarize(run_suite(Broken, [Case.from_dict(d) for d in json.loads(CASES.read_text())]))
    gate = compare(bad, baseline)
    assert not gate.ok
    assert any(r.startswith("critical case failed: refund-no-promise") for r in gate.reasons)
    assert any(r.startswith("regression: stock-in") for r in gate.reasons)
    assert any("pass rate dropped" in r for r in gate.reasons)
    assert compare(good, baseline).ok
    assert not compare(good, None, min_pass_rate=1.01).ok


def test_cli_exit_codes(tmp_path: Path) -> None:
    out, md = tmp_path / "r.json", tmp_path / "s.md"
    assert (
        main(["run", str(CASES), "--agent", "examples.demo_agent:make_agent", "--out", str(out), "--md", str(md)]) == 0
    )
    assert json.loads(out.read_text())["passed"] == 5
    assert "Gate passed" in md.read_text()
    assert (
        main(
            [
                "run",
                str(CASES),
                "--agent",
                "examples.demo_agent:make_agent",
                "--tags",
                "stock",
                "--min-pass-rate",
                "1.01",
            ]
        )
        == 1
    )


def test_cli_reports_bad_agent_specs_clearly() -> None:
    import pytest

    with pytest.raises(SystemExit, match=r"package\.module:factory"):
        main(["run", str(CASES), "--agent", "no_colon_here"])
    with pytest.raises(SystemExit, match="could not load agent factory"):
        main(["run", str(CASES), "--agent", "does.not.exist:make"])
