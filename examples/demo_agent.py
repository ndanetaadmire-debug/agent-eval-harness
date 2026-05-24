"""A tiny deterministic 'support agent' so the harness can be demoed and CI-tested without an API key.
Swap `make_agent` for an adapter around your real agent (LangGraph, OpenAI, HTTP)."""

from __future__ import annotations

import re

from evalh.model import ToolCall, Trace

STOCK = {"bob wig": 7, "body wave bundle": 0}
ORDERS = {"#1042": {"status": "shipped", "carrier": "The Courier Guy", "email": "thandi@example.com"}}


class DemoAgent:
    def run(self, messages: list[dict[str, str]]) -> Trace:
        text = messages[-1]["content"].lower()
        calls: list[ToolCall] = []
        if "ignore" in text and "instructions" in text:
            return Trace(reply="I can help with stock, orders and returns.", steps=1, input_tokens=40, output_tokens=12)
        if m := re.search(r"#?(\d{4})", text):
            order = ORDERS.get(f"#{m.group(1)}")
            calls.append(ToolCall("track_order", {"order": f"#{m.group(1)}"}, order, ok=order is not None))
            reply = (
                f"Order #{m.group(1)} has {order['status']} with {order['carrier']}."
                if order
                else "I couldn't find that order."
            )
        elif "refund" in text:
            calls.append(ToolCall("escalate_refund", {"reason": "refund_request"}, {"ticket": "ESC-1"}))
            reply = "I've passed your refund request to our team (ticket ESC-1)."
        else:
            item = next((k for k in STOCK if k in text), None)
            calls.append(ToolCall("check_stock", {"query": item}, {"available": STOCK.get(item or "", 0)}))
            reply = f"We have {STOCK.get(item or '', 0)} in stock." if item else "Which product do you mean?"
        return Trace(reply=reply, tool_calls=calls, steps=len(calls) + 1, input_tokens=120, output_tokens=30)


def make_agent() -> DemoAgent:
    return DemoAgent()
