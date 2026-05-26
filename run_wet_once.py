"""Run exactly one wet-experiment LGBO acquisition.

This runner does not evaluate the proposed points. It consumes already observed
experimental data, asks the LLM for a point/region preference, applies the LGBO
region-lifted BO step, and writes the next physical batch for lab execution.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from api_config import BATCH_Q, PRINT_LIMIT
from lgbo_core import ContinuousSpace, propose_lgbo_batch, serialize_plan, tensor_from_observations
from llm_client import call_chat
from prompt import build_user_prompt, choose_system_prompt, choose_user_prompt, parse_assistant_response


def read_text(path: str | None) -> str | None:
    return Path(path).read_text(encoding="utf-8") if path else None


FORMAT_GUARD = (
    "\n\n[Formatting Guard]\n"
    "- Write Final Answer FIRST, before any reasoning.\n"
    "- Final Answer must be exactly one strict bracketed structure: [point, [...], confidence] or [region, [[...], [...]], confidence].\n"
    "- After Final Answer, write at most three short Thinking bullets.\n"
    "- Do not include long chain-of-thought or extra prose before Final Answer.\n"
)

STRICT_FORMAT_GUARD = (
    "\n\n[Strict Formatting Retry]\n"
    "- Output ONLY the Final Answer line.\n"
    "- Required format: [point, [x1, x2, ..., xd], confidence] OR [region, [[lb1, ..., lbd], [ub1, ..., ubd]], confidence].\n"
)


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fallback_assistant(data: dict, space: ContinuousSpace) -> str:
    observations = data.get("observations", [])
    if observations:
        best = max(observations, key=lambda row: float(row.get(data.get("y_key", "y"), row.get("y"))))
        x = best.get("x", best.get("point"))
        if x:
            center = [float(x[name]) for name in space.names]
        else:
            center = [(lo + hi) * 0.5 for lo, hi in space.bounds]
    else:
        center = [(lo + hi) * 0.5 for lo, hi in space.bounds]
    return (
        "Thinking:\n"
        "For this offline smoke run, use the strongest observed setting as the semantic LGBO point preference.\n\n"
        "Final Answer:\n"
        f"[point, {center}, 0.55]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one wet-experiment LGBO acquisition.")
    parser.add_argument("--data-json", required=True)
    parser.add_argument("--batch-q", type=int)
    parser.add_argument("--policy", choices=["tilt", "cola"], default="tilt")
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--cand-size", type=int, default=2048)
    parser.add_argument("--num-paths-batch", type=int, default=256)
    parser.add_argument("--system-prompt")
    parser.add_argument("--system-prompt-file")
    parser.add_argument("--user-prompt")
    parser.add_argument("--user-prompt-file")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    data = load_json(args.data_json)
    parameters = list(data["parameters"])
    space = ContinuousSpace.from_parameters(parameters)
    y_key = str(data.get("y_key", "y"))
    goal = str(data.get("goal", "max")).lower()
    observations = list(data.get("observations", data.get("history", [])))
    X_hist, Y_hist = tensor_from_observations(observations, space, y_key=y_key)

    batch_q = int(args.batch_q or data.get("batch_q") or BATCH_Q)
    fallback_prompt = build_user_prompt(
        background=str(data.get("background", "")),
        parameters=parameters,
        objective=str(data.get("objective", "")),
        constraints=data.get("constraints", ""),
        history=observations,
        batch_q=batch_q,
        y_key=y_key,
        extra_request=data.get("extra_request"),
    )
    system_prompt = choose_system_prompt(args.system_prompt, read_text(args.system_prompt_file))
    user_prompt = choose_user_prompt(args.user_prompt, read_text(args.user_prompt_file), fallback_prompt)

    if args.offline:
        assistant_text = fallback_assistant(data, space)
    else:
        assistant_text = call_chat(system_prompt, user_prompt + FORMAT_GUARD)
    parsed = parse_assistant_response(assistant_text)
    if not parsed.get("mode") and not args.offline:
        assistant_text = call_chat(system_prompt, user_prompt + STRICT_FORMAT_GUARD)
        parsed = parse_assistant_response(assistant_text)
    if not parsed.get("mode"):
        preview = assistant_text if len(assistant_text) <= PRINT_LIMIT else assistant_text[:PRINT_LIMIT] + "\n... [truncated]"
        raise RuntimeError(f"Could not parse LLM preference. Raw output:\n{preview}")

    points, plan, Z_new = propose_lgbo_batch(
        X_norm=X_hist,
        y=Y_hist,
        parsed_preference=parsed,
        space=space,
        goal=goal,
        batch_q=batch_q,
        policy=args.policy,
        grid_size=args.grid_size,
        guidance_scale=args.guidance_scale,
        cand_size=args.cand_size,
        num_paths_batch=args.num_paths_batch,
        seed=args.seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"wet_once_{timestamp}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", *space.names])
        writer.writeheader()
        for idx, point in enumerate(points, start=1):
            writer.writerow({"index": idx, **point})
    json_path.write_text(
        json.dumps(
            {
                "mode": "wet",
                "source": args.data_json,
                "goal": goal,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "assistant_text": assistant_text,
                "parsed_preference": parsed,
                "lgbo_plan": serialize_plan(plan),
                "normalized_points": Z_new.tolist(),
                "points": points,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    preview = assistant_text if len(assistant_text) <= PRINT_LIMIT else assistant_text[:PRINT_LIMIT] + "\n... [truncated]"
    print(f"[wet] LGBO {args.policy} | goal={goal} batch_q={batch_q} guidance_scale={args.guidance_scale}")
    print(f"[wet] parsed={parsed.get('mode')} confidence={parsed.get('confidence')}")
    print(f"[wet] plan_mode={plan.get('mode')}")
    print(preview)
    print(f"[wet] wrote {csv_path}")
    print(f"[wet] wrote {json_path}")


if __name__ == "__main__":
    main()
