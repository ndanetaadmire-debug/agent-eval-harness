from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from evalh.model import CaseResult


@dataclass(slots=True)
class Summary:
    total: int
    passed: int
    pass_rate: float
    by_tag: dict[str, float]
    failing_checks: dict[str, int]
    p95_latency_ms: float
    mean_tokens: float
    cases: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "total": self.total,
                "passed": self.passed,
                "pass_rate": self.pass_rate,
                "by_tag": self.by_tag,
                "failing_checks": self.failing_checks,
                "p95_latency_ms": self.p95_latency_ms,
                "mean_tokens": self.mean_tokens,
                "cases": self.cases,
            },
            indent=2,
            sort_keys=True,
        )


def summarize(results: list[CaseResult]) -> Summary:
    tags: dict[str, list[bool]] = defaultdict(list)
    failing: dict[str, int] = defaultdict(int)
    latencies: list[float] = []
    tokens: list[int] = []
    cases: dict[str, dict[str, Any]] = {}
    for r in results:
        for t in r.tags or ["untagged"]:
            tags[t].append(r.passed)
        for f in r.failures:
            failing[f.name] += 1
        for tr in r.traces:
            latencies.append(tr.latency_ms)
            tokens.append(tr.input_tokens + tr.output_tokens)
        cases[r.case_id] = {
            "passed": r.passed,
            "pass_rate": r.pass_rate,
            "critical": r.critical,
            "failures": [f"{f.name}: {f.detail}" for f in r.failures],
        }
    passed = sum(r.passed for r in results)
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
    return Summary(
        total=len(results),
        passed=passed,
        pass_rate=round(passed / max(1, len(results)), 4),
        by_tag={k: round(sum(v) / len(v), 4) for k, v in sorted(tags.items())},
        failing_checks=dict(sorted(failing.items(), key=lambda kv: -kv[1])),
        p95_latency_ms=round(p95, 1),
        mean_tokens=round(sum(tokens) / max(1, len(tokens)), 1),
        cases=cases,
    )


@dataclass(frozen=True, slots=True)
class GateResult:
    ok: bool
    reasons: list[str]


def compare(
    current: Summary, baseline: dict[str, Any] | None, *, tolerance: float = 0.02, min_pass_rate: float = 0.0
) -> GateResult:
    """CI gate. Fails on: any failing critical case, a case that passed in baseline and now fails,
    an overall drop beyond tolerance, or falling under an absolute floor."""
    reasons: list[str] = []
    for cid, c in current.cases.items():
        if c["critical"] and not c["passed"]:
            reasons.append(f"critical case failed: {cid} ({'; '.join(c['failures'])})")
    if current.pass_rate < min_pass_rate:
        reasons.append(f"pass rate {current.pass_rate:.1%} below floor {min_pass_rate:.1%}")
    if baseline:
        if current.pass_rate < baseline["pass_rate"] - tolerance:
            reasons.append(f"pass rate dropped {baseline['pass_rate']:.1%} → {current.pass_rate:.1%}")
        for cid, prev in baseline.get("cases", {}).items():
            now = current.cases.get(cid)
            if prev["passed"] and now and not now["passed"]:
                reasons.append(f"regression: {cid} ({'; '.join(now['failures'])})")
    return GateResult(not reasons, reasons)


def to_markdown(s: Summary, gate: GateResult | None = None) -> str:
    lines = [
        "## Agent evaluation",
        "",
        f"**{s.passed}/{s.total} cases passed ({s.pass_rate:.1%})** · p95 latency {s.p95_latency_ms:.0f} ms · "
        f"mean tokens {s.mean_tokens:.0f}",
        "",
    ]
    if gate is not None:
        lines += [("✅ Gate passed" if gate.ok else "❌ Gate failed"), *[f"- {r}" for r in gate.reasons], ""]
    lines += ["| Tag | Pass rate |", "|---|---|", *[f"| {t} | {v:.0%} |" for t, v in s.by_tag.items()], ""]
    failed = {k: v for k, v in s.cases.items() if not v["passed"]}
    if failed:
        lines += ["### Failures", ""]
        lines += [
            f"- **{k}**{' (critical)' if v['critical'] else ''}: {'; '.join(v['failures'])}" for k, v in failed.items()
        ]
    return "\n".join(lines) + "\n"
