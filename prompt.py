"""Public prompt helpers for LGBO.

The paper-facing interface is intentionally strict: the LLM may only express a
preference as a point or as a hyper-rectangle plus confidence. The BO side then
turns that preference into a region-lifted surrogate update.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SYSTEM_PROMPT = """You are an optimization scientist inside an LLM-Guided Bayesian Optimization (LGBO) loop.

Your job is to provide one semantic preference for the next BO acquisition step.
The preference must be either a point or a compact axis-aligned region, plus a confidence score.
The downstream LGBO optimizer will convert your preference into a region-lifted mean shift and the acquisition function will select the actual batch.

Allowed final-answer formats:
1) [point, [x1, x2, ..., xd], confidence]
2) [region, [[lb1, lb2, ..., lbd], [ub1, ub2, ..., ubd]], confidence]

Rules:
- Pick exactly one mode: point or region.
- Use the declared parameter order exactly.
- Use original physical coordinates. Do not normalize.
- Keep all continuous values inside the declared bounds.
- confidence must be in [0, 1].
- Do not add any extra text inside Final Answer.

Response structure:
Thinking:
- Explain the scientific or function-structure rationale.
- Briefly relate the rationale to observed history.
- Justify point vs region and the confidence.

Final Answer:
[point, [x1, x2, ..., xd], confidence]
OR
[region, [[lb1, lb2, ..., lbd], [ub1, ub2, ..., ubd]], confidence]
"""


DEFAULT_USER_PROMPT_EXAMPLE = """[Background]
- Experiment type & purpose: Optimize a continuous experimental recipe.
- Parameter order (d=3): [temperature_c, catalyst_mol_pct, time_h]
- Parameter details:
  - temperature_c in [20, 90] C
  - catalyst_mol_pct in [0.1, 5.0] mol%
  - time_h in [0.5, 12.0] h
- Objective: Maximize product yield while avoiding unsafe operating conditions.
- Constraints:
  - temperature_c must stay below 90 C.
  - impurity should stay below 3%.

[History]
- temperature_c=55, catalyst_mol_pct=2.0, time_h=6.0 -> y=71
- temperature_c=72, catalyst_mol_pct=3.0, time_h=4.0 -> y=78

[Request]
- Recommend one point or one compact region for the next LGBO acquisition step.
"""


TOY_LANDSCAPES = {
    "rastrigin": (
        "Highly multimodal, separable, and periodic, with the global minimum near x=0."
    ),
    "ackley": (
        "Broad flat outer areas, ripples, and a central basin, with the global minimum near x=0."
    ),
    "griewank": (
        "Many shallow local minima from cosine interactions and a global minimum near x=0."
    ),
    "levy": (
        "Rugged with many local optima and the global minimum near x=1."
    ),
}


def _readable_bounds(param: Mapping[str, Any]) -> str:
    unit = f" {param.get('unit', '')}".rstrip()
    lo, hi = param["bounds"]
    return f"[{lo}, {hi}]{unit}"


def _format_history_item(item: Any, y_key: str = "y") -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        x = item.get("x", item.get("point", {}))
        y = item.get(y_key, item.get("y", item.get("result")))
        if isinstance(x, Mapping):
            left = ", ".join(f"{k}={v}" for k, v in x.items())
        else:
            left = str(x)
        if isinstance(y, Mapping):
            right = ", ".join(f"{k}={v}" for k, v in y.items())
        else:
            right = str(y)
        note = item.get("note")
        line = f"{left} -> {y_key}={right}" if right not in {"", "None"} else left
        if note:
            line += f" ({note})"
        return line
    return str(item)


def build_user_prompt(
    *,
    background: str,
    parameters: Sequence[Mapping[str, Any]],
    objective: str,
    constraints: str | Sequence[str] | None = None,
    history: Iterable[Any] | None = None,
    batch_q: int | None = None,
    y_key: str = "y",
    extra_request: str | None = None,
) -> str:
    names = [str(p["name"]) for p in parameters]
    parts: list[str] = []
    parts.append("[Background]")
    parts.append(f"- Experiment type & purpose: {background.strip()}")
    parts.append(f"- Parameter order (d={len(names)}): [" + ", ".join(names) + "]")
    parts.append("- Parameter details:")
    for param in parameters:
        desc = str(param.get("description", "")).strip()
        suffix = f"; {desc}" if desc else ""
        parts.append(f"  - {param['name']} in {_readable_bounds(param)}{suffix}")
    parts.append(f"- Objective: {objective.strip()}")

    if constraints:
        parts.append("- Constraints:")
        if isinstance(constraints, str):
            for line in constraints.strip().splitlines():
                if line.strip():
                    parts.append(f"  - {line.strip().lstrip('-').strip()}")
        else:
            for line in constraints:
                parts.append(f"  - {line}")

    hist = list(history or [])
    if hist:
        parts.append("")
        parts.append("[History]")
        for item in hist:
            parts.append(f"- {_format_history_item(item, y_key=y_key)}")

    parts.append("")
    parts.append("[Request]")
    if batch_q:
        parts.append(
            f"- Recommend one point or one compact region. LGBO will draw {batch_q} batch points from the lifted surrogate."
        )
    else:
        parts.append("- Recommend one point or one compact region for the next LGBO acquisition step.")
    if extra_request:
        parts.append(f"- {extra_request.strip()}")
    return "\n".join(parts)


def build_toy_user_prompt(
    *,
    func_name: str,
    d: int,
    bounds: Sequence[Sequence[float]],
    history: Iterable[Any] | None,
    batch_q: int,
) -> str:
    name = func_name.strip().lower()
    if name not in TOY_LANDSCAPES:
        raise ValueError("func_name must be one of: rastrigin, ackley, griewank, levy")
    parameters = [
        {
            "name": f"x{i + 1}",
            "bounds": [float(lo), float(hi)],
            "unit": "",
            "description": "continuous toy-function coordinate",
        }
        for i, (lo, hi) in enumerate(bounds)
    ]
    return build_user_prompt(
        background=f"Dry benchmark on the {name} black-box function. Landscape: {TOY_LANDSCAPES[name]}",
        parameters=parameters,
        objective="Minimize f(x). Smaller f is better.",
        constraints="Use original toy-function coordinates and stay inside bounds.",
        history=history,
        batch_q=batch_q,
        y_key="f",
        extra_request="Return original coordinates, not normalized coordinates.",
    )


_THINKING_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:#+\s*)?\[?\s*thinking\s*\]?\s*:?\s*(.*?)"
    r"(?=\n\s*(?:#+\s*)?\[?\s*final\s+answer\s*\]?\s*:|\Z)"
)
_POINT_RE = re.compile(
    r"(?is)\[\s*point\s*,\s*(\[[^\[\]]*\])\s*,\s*([01](?:\.\d+)?)\s*\]"
)
_REGION_RE = re.compile(
    r"(?is)\[\s*region\s*,\s*\[\s*(\[[^\[\]]*\])\s*,\s*(\[[^\[\]]*\])\s*\]\s*,\s*([01](?:\.\d+)?)\s*\]"
)


def _parse_list(text: str) -> list[Any]:
    value = ast.literal_eval(text)
    if not isinstance(value, list):
        raise ValueError("expected list")
    return value


def parse_assistant_response(assistant_text: str) -> dict[str, Any]:
    text = assistant_text or ""
    thinking = ""
    m_thinking = _THINKING_RE.search(text)
    if m_thinking:
        thinking = m_thinking.group(1).strip()

    m_region = _REGION_RE.search(text)
    if m_region:
        try:
            return {
                "mode": "region",
                "lb": _parse_list(m_region.group(1)),
                "ub": _parse_list(m_region.group(2)),
                "confidence": float(m_region.group(3)),
                "thinking": thinking,
            }
        except Exception:
            pass

    m_point = _POINT_RE.search(text)
    if m_point:
        try:
            return {
                "mode": "point",
                "point": _parse_list(m_point.group(1)),
                "confidence": float(m_point.group(2)),
                "thinking": thinking,
            }
        except Exception:
            pass

    return {"mode": None, "confidence": None, "thinking": thinking}


_parse_assistant = parse_assistant_response


def choose_system_prompt(user_value: str | None, file_value: str | None) -> str:
    if user_value and user_value.strip():
        return user_value.strip()
    if file_value and file_value.strip():
        return file_value.strip()
    return DEFAULT_SYSTEM_PROMPT


def choose_user_prompt(user_value: str | None, file_value: str | None, fallback: str | None) -> str:
    if user_value and user_value.strip():
        return user_value.strip()
    if file_value and file_value.strip():
        return file_value.strip()
    return (fallback or DEFAULT_USER_PROMPT_EXAMPLE).strip()
