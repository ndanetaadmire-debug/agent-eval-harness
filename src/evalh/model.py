from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    ok: bool = True


@dataclass(slots=True)
class Trace:
    """What an agent did for one case: the reply plus everything needed to judge *how* it got there."""

    reply: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class Agent(Protocol):
    """Adapter around the system under test (LangGraph, LangChain, a raw API loop, an HTTP endpoint…)."""

    def run(self, messages: list[dict[str, str]]) -> Trace: ...


ToolMatch = Literal["exact", "ordered", "unordered", "none"]


@dataclass(slots=True)
class Expectation:
    tools: list[str] = field(default_factory=list)
    tool_match: ToolMatch = "ordered"
    forbidden_tools: list[str] = field(default_factory=list)
    tool_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    grounded_numbers: bool = False
    no_pii_leak: bool = True
    max_steps: int | None = None
    max_latency_ms: float | None = None
    max_tokens: int | None = None
    rubric: str | None = None


@dataclass(slots=True)
class Case:
    id: str
    messages: list[dict[str, str]]
    expect: Expectation
    tags: list[str] = field(default_factory=list)
    critical: bool = False
    repeats: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Case:
        msgs = d.get("messages") or [{"role": "user", "content": d["input"]}]
        return cls(
            id=d["id"],
            messages=msgs,
            expect=Expectation(**d.get("expect", {})),
            tags=d.get("tags", []),
            critical=d.get("critical", False),
            repeats=d.get("repeats", 1),
        )


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class CaseResult:
    case_id: str
    tags: list[str]
    critical: bool
    runs: list[list[Check]]
    traces: list[Trace]

    @property
    def pass_rate(self) -> float:
        return sum(all(c.passed for c in run) for run in self.runs) / len(self.runs)

    @property
    def passed(self) -> bool:
        """A case passes only if *every* repeat passes: flaky agents are failing agents."""
        return self.pass_rate == 1.0

    @property
    def failures(self) -> list[Check]:
        seen: dict[str, Check] = {}
        for run in self.runs:
            for c in run:
                if not c.passed and c.name not in seen:
                    seen[c.name] = c
        return list(seen.values())
