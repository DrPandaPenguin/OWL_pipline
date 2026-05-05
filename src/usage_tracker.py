# openAI API 호출량/비용 추적용. install()로 한 번 monkey-patch 걸어두면 자동 집계됨
import time
from collections import defaultdict


# 1M 토큰당 USD (input/output 분리) — 가격은 OpenAI 페이지 참고
_PRICES_PER_M = {
    "gpt-5.2":     {"in": 1.25, "out": 10.00},
    "gpt-5.4":     {"in": 1.25, "out": 10.00},
    "gpt-5":       {"in": 1.25, "out": 10.00},
    "gpt-5-mini":  {"in": 0.25, "out": 2.00},
    "gpt-5-nano":  {"in": 0.05, "out": 0.40},
    "gpt-4.5":     {"in": 75.00, "out": 150.00},
    "gpt-4.1":     {"in": 2.00,  "out": 8.00},
    "gpt-4o":      {"in": 2.50,  "out": 10.00},
    "gpt-4o-mini": {"in": 0.15,  "out": 0.60},
    "gpt-4-turbo": {"in": 10.00, "out": 30.00},
    "gpt-4":       {"in": 30.00, "out": 60.00},
}


def _price_for(model):
    if model in _PRICES_PER_M:
        return _PRICES_PER_M[model]
    for prefix in sorted(_PRICES_PER_M.keys(), key=len, reverse=True):
        if model.startswith(prefix):
            return _PRICES_PER_M[prefix]
    return _PRICES_PER_M["gpt-5.2"]


_state = {
    "calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "by_model": defaultdict(lambda: {"calls": 0, "input": 0, "output": 0}),
    "started_at": None,
    "ended_at": None,
}


def reset():
    _state["calls"] = 0
    _state["input_tokens"] = 0
    _state["output_tokens"] = 0
    _state["by_model"] = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0})
    _state["started_at"] = time.time()
    _state["ended_at"] = None


def _record(model, prompt_tokens, completion_tokens):
    _state["calls"] += 1
    _state["input_tokens"] += prompt_tokens
    _state["output_tokens"] += completion_tokens
    b = _state["by_model"][model]
    b["calls"] += 1
    b["input"] += prompt_tokens
    b["output"] += completion_tokens


def snapshot():
    _state["ended_at"] = _state["ended_at"] or time.time()
    total_cost = 0.0
    by_model = {}
    for model, b in _state["by_model"].items():
        p = _price_for(model)
        cost = (b["input"] / 1_000_000) * p["in"] + (b["output"] / 1_000_000) * p["out"]
        total_cost += cost
        by_model[model] = {
            **b,
            "cost_usd": round(cost, 4),
            "price_in_per_M": p["in"],
            "price_out_per_M": p["out"],
        }
    elapsed = (_state["ended_at"] or time.time()) - (_state["started_at"] or time.time())
    return {
        "calls": _state["calls"],
        "input_tokens": _state["input_tokens"],
        "output_tokens": _state["output_tokens"],
        "total_cost_usd": round(total_cost, 4),
        "elapsed_s": round(elapsed, 2),
        "by_model": by_model,
    }


def report():
    s = snapshot()
    lines = [
        "--- LLM usage ---",
        f" Calls:  {s['calls']}",
        f" Tokens: {s['input_tokens']:,} in / {s['output_tokens']:,} out  (total {s['input_tokens'] + s['output_tokens']:,})",
        f" Cost:   ${s['total_cost_usd']:.4f}",
        f" Wall:   {s['elapsed_s']:.1f}s",
    ]
    if len(s["by_model"]) > 1 or (s["by_model"] and list(s["by_model"].keys())[0] != "gpt-5.2"):
        lines.append(" Per-model:")
        for model, b in s["by_model"].items():
            lines.append(
                f"   {model:18s}  {b['calls']:3d} calls  "
                f"{b['input']:>9,} in / {b['output']:>7,} out  ${b['cost_usd']:.4f}"
            )
    return "\n".join(lines)


_INSTALLED = False


# openAI SDK의 chat.completions.create를 감싸서 호출 시점에 토큰 자동 집계
def install():
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from openai.resources.chat.completions import Completions
    except Exception:
        try:
            from openai.resources.chat import Completions
        except Exception:
            return

    orig = Completions.create

    def patched(self, *args, **kwargs):
        resp = orig(self, *args, **kwargs)
        try:
            u = getattr(resp, "usage", None)
            if u is not None:
                model = getattr(resp, "model", None) or kwargs.get("model") or "unknown"
                _record(
                    str(model),
                    int(getattr(u, "prompt_tokens", 0) or 0),
                    int(getattr(u, "completion_tokens", 0) or 0),
                )
        except Exception:
            pass
        return resp

    Completions.create = patched
    _INSTALLED = True
