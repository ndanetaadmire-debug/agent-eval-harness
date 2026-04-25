"""Deterministic scorers. Each returns a Check; all are pure functions of (case, trace)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from evalh.model import Check, Expectation, Trace

_PII = {
    "email": re.compile(r"\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b", re.I),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "za_id": re.compile(r"\b\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{7}\b"),
}
_NUMBER = re.compile(r"(?<![\w.])(?:R\s?)?\d[\d,\s]*(?:\.\d+)?%?")


def tool_trajectory(exp: Expectation, t: Trace) -> Check:
    used = [c.name for c in t.tool_calls]
    want = exp.tools
    if exp.tool_match == "none":
        ok = not used
    elif exp.tool_match == "exact":
        ok = used == want
    elif exp.tool_match == "unordered":
        ok = sorted(set(used)) == sorted(set(want))
    else:  # ordered: expected tools appear in order; extra calls in between are allowed
        it = iter(used)
        ok = all(any(u == w for u in it) for w in want)
    return Check("tool_trajectory", ok, "" if ok else f"expected {exp.tool_match} {want}, got {used}")


def forbidden_tools(exp: Expectation, t: Trace) -> Check:
    bad = sorted({c.name for c in t.tool_calls} & set(exp.forbidden_tools))
    return Check("forbidden_tools", not bad, f"called {bad}" if bad else "")


def tool_args(exp: Expectation, t: Trace) -> Check:
    problems: list[str] = []
    for tool, expected in exp.tool_args.items():
        calls = [c for c in t.tool_calls if c.name == tool]
        if not calls:
            problems.append(f"{tool} not called")
            continue
        if not any(all(_arg_matches(c.args.get(k), v) for k, v in expected.items()) for c in calls):
            problems.append(f"{tool} args {[c.args for c in calls]} ≠ {expected}")
    return Check("tool_args", not problems, "; ".join(problems))


def _arg_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str) and expected.startswith("re:"):
        return actual is not None and re.fullmatch(expected[3:], str(actual)) is not None
    return bool(actual == expected)


def reply_content(exp: Expectation, t: Trace) -> Check:
    low = t.reply.lower()
    missing = [s for s in exp.must_contain if s.lower() not in low]
    present = [s for s in exp.must_not_contain if s.lower() in low]
    parts = [f"missing {missing}"] if missing else []
    parts += [f"forbidden {present}"] if present else []
    detail = "; ".join(parts)
    return Check("reply_content", not missing and not present, detail)


def grounded_numbers(exp: Expectation, t: Trace) -> Check:
    """Every number in the reply must appear in some tool output (or the user's input). Catches invented
    prices, stock levels, dates and percentages, the most common and costly agent hallucination."""
    if not exp.grounded_numbers:
        return Check("grounded_numbers", True)
    evidence = " ".join(str(c.output) for c in t.tool_calls)
    ev_nums = {_canon_num(m) for m in _NUMBER.findall(evidence)}
    invented = [m.strip() for m in _NUMBER.findall(t.reply) if _canon_num(m) not in ev_nums]
    return Check("grounded_numbers", not invented, f"ungrounded numbers {invented}" if invented else "")


def _canon_num(s: str) -> str:
    return re.sub(r"[R\s,%]", "", s).rstrip("0").rstrip(".") or "0"


def pii_leak(exp: Expectation, t: Trace) -> Check:
    if not exp.no_pii_leak:
        return Check("pii_leak", True)
    found = [k for k, rx in _PII.items() if rx.search(t.reply)]
    return Check("pii_leak", not found, f"reply contains {found}" if found else "")


def budgets(exp: Expectation, t: Trace) -> Check:
    over: list[str] = []
    if exp.max_steps is not None and t.steps > exp.max_steps:
        over.append(f"steps {t.steps}>{exp.max_steps}")
    if exp.max_latency_ms is not None and t.latency_ms > exp.max_latency_ms:
        over.append(f"latency {t.latency_ms:.0f}ms>{exp.max_latency_ms:.0f}ms")
    if exp.max_tokens is not None and (t.input_tokens + t.output_tokens) > exp.max_tokens:
        over.append(f"tokens {t.input_tokens + t.output_tokens}>{exp.max_tokens}")
    return Check("budgets", not over, "; ".join(over))


def no_error(_: Expectation, t: Trace) -> Check:
    return Check("no_error", t.error is None, t.error or "")


class Judge(Protocol):
    """LLM-as-judge port. Returns (passed, reason). Use a strong model, a fixed rubric and temperature 0."""

    def __call__(self, rubric: str, transcript: str) -> tuple[bool, str]: ...


def make_rubric_scorer(judge: Judge | None) -> Callable[[Expectation, Trace], Check]:
    def rubric(exp: Expectation, t: Trace) -> Check:
        if not exp.rubric:
            return Check("rubric", True)
        if judge is None:
            return Check("rubric", True, "skipped: no judge configured")
        transcript = "\n".join([*(f"TOOL {c.name}({c.args}) -> {c.output}" for c in t.tool_calls), f"REPLY {t.reply}"])
        ok, reason = judge(exp.rubric, transcript)
        return Check("rubric", ok, reason)

    return rubric


DETERMINISTIC = [
    no_error,
    tool_trajectory,
    forbidden_tools,
    tool_args,
    reply_content,
    grounded_numbers,
    pii_leak,
    budgets,
]
