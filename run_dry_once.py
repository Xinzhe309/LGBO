"""Run dry LGBO acquisitions on toy functions."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.quasirandom import SobolEngine

from api_config import BATCH_Q, N_INIT, PRINT_LIMIT
from fun.toy_fun import get_bo_bounds, toy_results
from lgbo_core import ContinuousSpace, propose_lgbo_batch, serialize_plan
from llm_client import call_chat
from prompt import build_toy_user_prompt, choose_system_prompt, choose_user_prompt, parse_assistant_response


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


def toy_parameters(func_name: str, d: int) -> list[dict[str, Any]]:
    return [{"name": f"x{i + 1}", "bounds": list(bounds)} for i, bounds in enumerate(get_bo_bounds(func_name, d))]


def make_initial_data(func_name: str, space: ContinuousSpace, n_init: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    sobol = SobolEngine(dimension=space.d, scramble=True, seed=seed)
    X = sobol.draw(max(2, int(n_init))).to(dtype=torch.double).clamp(1e-6, 1.0 - 1e-6)
    physical = [space.denormalize_vector(row.tolist()) for row in X]
    y_values = toy_results(func_name, physical)
    Y = torch.tensor([[float(row["f"])] for row in y_values], dtype=torch.double)
    history = []
    for x, y in zip(physical, y_values):
        left = ", ".join(f"{name}={value:.6g}" for name, value in zip(space.names, x))
        history.append(f"{left} -> f={float(y['f']):.6g}")
    return X, Y, history


def read_history_csv(path: str, space: ContinuousSpace) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8-sig").splitlines()))
    X_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    history: list[str] = []
    for row in rows:
        x = [float(row[name]) for name in space.names]
        f = float(row["f"])
        X_rows.append(space.normalize_vector(x))
        y_rows.append([f])
        left = ", ".join(f"{name}={value:g}" for name, value in zip(space.names, x))
        history.append(f"{left} -> f={f:g}")
    if len(X_rows) < 2:
        raise ValueError("history CSV must contain at least two rows")
    return torch.tensor(X_rows, dtype=torch.double), torch.tensor(y_rows, dtype=torch.double), history


def fallback_assistant(func_name: str, d: int) -> str:
    center = [1.0] * d if func_name == "levy" else [0.0] * d
    return (
        "Thinking:\n"
        "For this offline smoke run, use a known promising dry-benchmark basin as the semantic LGBO point preference.\n\n"
        "Final Answer:\n"
        f"[point, {center}, 0.65]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dry LGBO acquisitions on a toy function.")
    parser.add_argument("--func", default="ackley", choices=["rastrigin", "ackley", "griewank", "levy"])
    parser.add_argument("--dim", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--batch-q", type=int, default=BATCH_Q)
    parser.add_argument("--n-init", type=int, default=N_INIT)
    parser.add_argument("--history-csv")
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    parameters = toy_parameters(args.func, args.dim)
    space = ContinuousSpace.from_parameters(parameters)
    if args.history_csv:
        X_hist, Y_hist, history = read_history_csv(args.history_csv, space)
    else:
        X_hist, Y_hist, history = make_initial_data(args.func, space, args.n_init, args.seed)
    goal = "min"
    y_key = "f"
    stem_base = f"dry_{args.func}_d{args.dim}"
    task_dim = args.dim

    system_prompt = choose_system_prompt(args.system_prompt, read_text(args.system_prompt_file))

    rows = []
    round_payloads = []
    best_value = float(Y_hist.max().item()) if goal == "max" else float(Y_hist.min().item())
    best_x = None

    for round_id in range(1, max(1, int(args.rounds)) + 1):
        fallback_prompt = build_toy_user_prompt(
            func_name=args.func,
            d=args.dim,
            bounds=space.bounds,
            history=history,
            batch_q=args.batch_q,
        )

        user_prompt = choose_user_prompt(args.user_prompt, read_text(args.user_prompt_file), fallback_prompt)

        if args.offline:
            assistant_text = fallback_assistant(args.func, space.d)
        else:
            assistant_text = call_chat(system_prompt, user_prompt + FORMAT_GUARD)
        parsed = parse_assistant_response(assistant_text)
        if not parsed.get("mode") and not args.offline:
            assistant_text = call_chat(system_prompt, user_prompt + STRICT_FORMAT_GUARD)
            parsed = parse_assistant_response(assistant_text)
        if not parsed.get("mode"):
            preview = assistant_text if len(assistant_text) <= PRINT_LIMIT else assistant_text[:PRINT_LIMIT] + "\n... [truncated]"
            raise RuntimeError(f"Could not parse LLM preference at round {round_id}. Raw output:\n{preview}")

        points, plan, Z_new = propose_lgbo_batch(
            X_norm=X_hist,
            y=Y_hist,
            parsed_preference=parsed,
            space=space,
            goal=goal,
            batch_q=args.batch_q,
            policy=args.policy,
            grid_size=args.grid_size,
            guidance_scale=args.guidance_scale,
            cand_size=args.cand_size,
            num_paths_batch=args.num_paths_batch,
            seed=args.seed + round_id,
        )

        vectors = [[point[name] for name in space.names] for point in points]
        results = toy_results(args.func, vectors)
        y_new = torch.tensor([[float(result["f"])] for result in results], dtype=torch.double)
        new_history = []
        for idx, (point, result) in enumerate(zip(points, results), start=1):
            score = float(result["f"])
            if score <= best_value:
                best_value = score
                best_x = point
            left = ", ".join(f"{name}={point[name]:.6g}" for name in space.names)
            new_history.append(f"{left} -> f={score:.6g}")
            rows.append({"round": round_id, "index": idx, **point, "f": score, "best_f": best_value})

        X_new = torch.tensor([space.normalize_point(point) for point in points], dtype=torch.double)
        X_hist = torch.cat([X_hist, X_new], dim=0)
        Y_hist = torch.cat([Y_hist, y_new], dim=0)
        history = new_history + history

        round_payloads.append({
            "round": round_id,
            "assistant_text": assistant_text,
            "parsed_preference": parsed,
            "lgbo_plan": serialize_plan(plan),
            "normalized_points": Z_new.tolist(),
        })
        print(
            f"[dry:r{round_id}] parsed={parsed.get('mode')} conf={parsed.get('confidence')} "
            f"plan={plan.get('mode')} best_{y_key}={best_value:.6g}"
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stem_base}_r{max(1, int(args.rounds))}_{timestamp}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["round", "index", *space.names, y_key]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "mode": "dry",
                "function": args.func,
                "dimension": task_dim,
                "rounds": max(1, int(args.rounds)),
                "system_prompt": system_prompt,
                "rounds_detail": round_payloads,
                "points": rows,
                "best_value": best_value,
                "best_x": best_x,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[dry] LGBO {args.policy} | function={args.func} d={task_dim} rounds={max(1, int(args.rounds))} batch_q={args.batch_q} guidance_scale={args.guidance_scale}")
    print(f"[dry] best_{y_key}={best_value:.6g}")
    print(f"[dry] wrote {csv_path}")
    print(f"[dry] wrote {json_path}")


if __name__ == "__main__":
    main()
