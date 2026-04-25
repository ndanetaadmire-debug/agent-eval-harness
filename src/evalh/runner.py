from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from evalh.model import Agent, Case, CaseResult, Check, Expectation, Trace
from evalh.scorers import DETERMINISTIC, Judge, make_rubric_scorer

Scorer = Callable[[Expectation, Trace], Check]


def run_case(agent_factory: Callable[[], Agent], case: Case, scorers: Sequence[Scorer]) -> CaseResult:
    """Run a case `repeats` times with a *fresh* agent each time (no state leaking between runs)."""
    runs: list[list[Check]] = []
    traces: list[Trace] = []
    for _ in range(max(1, case.repeats)):
        start = time.perf_counter()
        try:
            trace = agent_factory().run(case.messages)
        except Exception as exc:  # an exception is a failed run, not a crashed suite
            trace = Trace(reply="", error=f"{type(exc).__name__}: {exc}")
        if not trace.latency_ms:
            trace.latency_ms = (time.perf_counter() - start) * 1000
        traces.append(trace)
        runs.append([s(case.expect, trace) for s in scorers])
    return CaseResult(case.id, case.tags, case.critical, runs, traces)


def run_suite(
    agent_factory: Callable[[], Agent],
    cases: Sequence[Case],
    *,
    judge: Judge | None = None,
    concurrency: int = 4,
    extra_scorers: Sequence[Scorer] = (),
) -> list[CaseResult]:
    scorers = [*DETERMINISTIC, make_rubric_scorer(judge), *extra_scorers]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(lambda c: run_case(agent_factory, c, scorers), cases))
