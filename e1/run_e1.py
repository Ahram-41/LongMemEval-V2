#!/usr/bin/env python3
"""E1 driver: resumable LongMemEval-V2 run.

Phase A builds the shared memory index once per (method, domain, tier) under
--index-root and saves it with the official --save-memory/--skip-evaluation path.
Phase B evaluates with --load-memory-dir; reader outputs are checkpointed per
question (reader_outputs_cache.jsonl, see evaluation/harness.py), so a job that
hits the Slurm time limit is simply re-submitted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation import run_eval as RE  # noqa: E402


def log(msg: str) -> None:
    print(f"[run_e1] {msg}", flush=True)


def harness_base_argv(args: argparse.Namespace, runtime_dir: Path, data_root: Path) -> list[str]:
    return [
        sys.executable, "-m", "evaluation.harness",
        "--domain", args.domain,
        "--questions-path", str(runtime_dir / "questions.json"),
        "--haystack-path", str(runtime_dir / "haystack.json"),
        "--trajectories-path", str(data_root / "trajectories.jsonl"),
        "--memory-config-path", str(runtime_dir / "memory_config.json"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-root", required=True, help="Directory holding reusable memory indexes")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--index-only", action="store_true", help="Build the index and stop")
    ap.add_argument("--smoke-trajectories", type=int, default=None,
                    help="Smoke test: keep only the first N trajectory ids of the haystack (index stored under a separate name)")
    own, passthrough = ap.parse_known_args()

    # Reuse run_eval's argparse so every official default/env var applies.
    sys.argv = ["run_eval.py", *passthrough]
    args = RE.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runtime_dir = output_dir / "runtime_inputs"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get(args.evaluator_api_key_env, "") == "":
        log(f"WARNING: {args.evaluator_api_key_env} is empty; LLM-judge scoring will fail after reader outputs are cached.")
        os.environ.setdefault(args.evaluator_api_key_env, "EMPTY")
    os.environ.setdefault(args.reader_api_key_env, "EMPTY")

    selected = RE.materialize_runtime_questions(
        data_root=data_root, domain=args.domain,
        question_ids=RE.parse_question_ids(args.question_ids), limit=args.limit,
        output_path=runtime_dir / "questions.json",
    )
    RE.materialize_runtime_haystack(
        data_root=data_root, tier=args.tier, selected_questions=selected,
        output_path=runtime_dir / "haystack.json",
    )
    if own.smoke_trajectories:
        hs_path = runtime_dir / "haystack.json"
        hs = json.loads(hs_path.read_text())
        hs = {q: ids[: own.smoke_trajectories] for q, ids in hs.items()}
        hs_path.write_text(json.dumps(hs, indent=2) + "\n")
        log(f"SMOKE: haystack truncated to {own.smoke_trajectories} trajectories per question")
    memory_config = RE.build_memory_config(args, data_root)
    RE.write_json(runtime_dir / "memory_config.json", memory_config)
    log(f"{len(selected)} questions, method={args.method} domain={args.domain} tier={args.tier}")

    # ---- Phase A: shared index -------------------------------------------------
    index_name = f"{args.method}_{args.domain}_{args.tier}" + (f"_smoke{own.smoke_trajectories}" if own.smoke_trajectories else "")
    index_dir = Path(own.index_root).expanduser().resolve() / index_name
    memory_state = index_dir / "memory_state"
    complete = (memory_state / "index.json").exists() and (memory_state / "memory_config.json").exists()
    if own.rebuild_index and index_dir.exists():
        log(f"--rebuild-index: removing {index_dir}")
        shutil.rmtree(index_dir)
        complete = False
    if not complete and index_dir.exists():
        log(f"partial index found, removing {index_dir}")
        shutil.rmtree(index_dir)
    if not complete:
        log(f"building index -> {memory_state}")
        cmd = harness_base_argv(args, runtime_dir, data_root) + [
            "--output-dir", str(index_dir),
            "--save-memory", "--skip-evaluation",
            "--prompt-build-max-workers", "1",
            "--memory-context-max-tokens", str(args.memory_context_max_tokens),
        ]
        subprocess.run(cmd, check=True, cwd=REPO)
        (index_dir / "INDEX_COMPLETE").write_text("ok\n")
    else:
        log(f"reusing index {memory_state}")
    if own.index_only:
        return

    # ---- Phase B: evaluation (resumable) ----------------------------------------
    if (output_dir / "aggregated_metrics.json").exists():
        log(f"already complete: {output_dir / 'aggregated_metrics.json'}")
        return
    cmd = harness_base_argv(args, runtime_dir, data_root) + [
        "--output-dir", str(output_dir),
        "--load-memory-dir", str(memory_state),
        "--model", args.reader_model,
        "--base-url", args.reader_base_url,
        "--api-key-env", args.reader_api_key_env,
        "--temperature", str(args.reader_temperature),
        "--top-p", str(args.reader_top_p),
        "--top-k", str(args.reader_top_k),
        "--max-completion-tokens", str(args.max_completion_tokens),
        "--memory-context-max-tokens", str(args.memory_context_max_tokens),
        "--reader-max-concurrent-requests", str(args.reader_max_concurrent_requests),
        "--prompt-build-max-workers", str(args.prompt_build_max_workers),
        "--evaluator-model", args.evaluator_model,
        "--evaluator-api-key-env", args.evaluator_api_key_env,
        "--evaluator-reasoning-effort", args.evaluator_reasoning_effort,
        "--evaluator-max-completion-tokens", str(args.evaluator_max_completion_tokens),
    ]
    if not args.reader_enable_thinking:
        cmd.append("--reader-disable-thinking")
    if args.shuffle_questions_seed is not None:
        cmd += ["--shuffle-questions-seed", str(args.shuffle_questions_seed)]
    # prompt_rows.jsonl / run_args.json from an interrupted attempt are rewritten; cache is appended.
    subprocess.run(cmd, check=True, cwd=REPO)
    metrics = json.loads((output_dir / "aggregated_metrics.json").read_text())
    log("overall_full_set = %s" % metrics.get("overall_full_set", metrics.get("overall")))


if __name__ == "__main__":
    main()
