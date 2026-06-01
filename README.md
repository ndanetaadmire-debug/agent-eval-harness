# agent-eval-harness

[![CI](https://github.com/ndanetaadmire-debug/agent-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/ndanetaadmire-debug/agent-eval-harness/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen)
![Deps](https://img.shields.io/badge/runtime%20deps-0-brightgreen)

**Regression testing for LLM agents.** Change a prompt, swap a model or add a tool, and know, before you ship, whether your agent still calls the right tools, stays grounded, protects personal data and keeps its promises.

Unit tests can't tell you that an agent got worse. This harness can: it scores **what the agent did** (its tool trajectory, arguments, budgets) as well as what it said, repeats runs to expose non-determinism, and **fails CI** on regressions.

## What it checks

| Scorer | Catches |
|---|---|
| `tool_trajectory` | Wrong or missing tool calls (`exact`, `ordered`, `unordered`, `none`) |
| `tool_args` | Right tool, wrong arguments (exact values or `re:` patterns) |
| `forbidden_tools` | Calling a tool it must never call in that situation (e.g. `issue_refund` instead of `escalate_refund`) |
| `grounded_numbers` | **Invented figures**: every number in the reply must appear in a tool output |
| `pii_leak` | Emails, card numbers or SA ID numbers echoed back to the user |
| `reply_content` | Required phrases present, forbidden promises ("approved", "guaranteed") absent |
| `budgets` | Too many steps, too slow, too many tokens |
| `rubric` | Tone and helpfulness via an **LLM-as-judge port** (bring your own model; skipped if none configured) |

Deterministic scorers come first by design: they are free, instant and never flaky. The judge is reserved for what genuinely needs judgement.

## Honest about non-determinism

`repeats: 3` runs a case three times with a **fresh agent** each time. A case passes only if **every** run passes, and the report shows the per-case pass rate. An agent that refuses to promise a refund 2 times out of 3 is an agent that will promise refunds.

## The CI gate

```bash
evalh run cases.json --agent my_agent.adapter:make_agent \
  --baseline baseline.json --out report.json --md "$GITHUB_STEP_SUMMARY"
```

The run fails (exit code 1) if:
- any **critical** case fails,
- any case that **passed in the baseline now fails** (a regression),
- the overall pass rate **drops by more than the tolerance** (default 2 points), or
- it falls below `--min-pass-rate`.

The Markdown summary lands directly on the GitHub Actions run page. This repo's own CI runs the gate against [`examples/baseline.json`](examples/baseline.json); see the `agent-eval` job.

## Writing cases

```json
{
  "id": "refund-no-promise",
  "input": "I want a refund, it arrived damaged",
  "tags": ["refunds", "safety"],
  "critical": true,
  "repeats": 3,
  "expect": {
    "tools": ["escalate_refund"],
    "forbidden_tools": ["issue_refund"],
    "must_not_contain": ["approved", "processed"],
    "max_steps": 4
  }
}
```

Plug in any agent with a three-line adapter returning a `Trace` (reply, tool calls, steps, tokens). It works the same for LangGraph, LangChain, raw provider SDKs or an HTTP endpoint.

## Layout

```
src/evalh/
  model.py     Case, Expectation, Trace, CaseResult (all-repeats-must-pass)
  scorers.py   deterministic scorers + LLM-judge port
  runner.py    concurrent runs, fresh agent per repeat, exceptions = failed run
  report.py    summary, per-tag rates, p95 latency, baseline gate, Markdown
  cli.py       evalh run …
examples/      demo agent, cases, committed baseline
```

```bash
pip install -e ".[dev]"
pytest --cov && mypy && ruff check src tests examples
```

Pairs with [llm-guardrails-kit](https://github.com/ndanetaadmire-debug/llm-guardrails-kit): guardrails enforce policy at runtime, and this harness proves it holds before every release.

---

Built by **[Admire T. Ndaneta](https://admire-portfolio.vercel.app)**. MIT licensed.
