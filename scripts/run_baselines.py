#!/usr/bin/env python3
"""Run pretrained PedROAM baselines, reference audit, and continuous 10 m alignment."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = "alignment-10m-continuous"
LABELS = {"ronin": "Madgwick", "tlio": "TLIO-EKF"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Dataset directory containing recordings/")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="New results directory; existing directories are refused")
    parser.add_argument("--model", choices=["ronin", "tlio", "both"], default="both")
    parser.add_argument("--checkpoint-root", type=Path,
                        default=ROOT.parent / "outputs/eqnio-pretrained",
                        help="Directory containing Ronin_o2/ and/or TLIO_o2/")
    parser.add_argument("--session", help="Optional recording session, e.g. 0423-PM-2026")
    parser.add_argument("--person", help="Optional phone/person ID, e.g. P02")
    parser.add_argument("--max-seconds", type=float, help="Process only the first N seconds per recording")
    parser.add_argument("--workers", type=int, default=4, help="TLIO worker processes (default: 4)")
    parser.add_argument("--initialization-distance-m", type=float, default=10.,
                        help="Shared initial GT travel distance for both inference and evaluation")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for both models")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate input paths and print commands without creating results")
    args = parser.parse_args()
    for field in ["data_root", "out_root", "checkpoint_root"]:
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not math.isfinite(args.initialization_distance_m) or args.initialization_distance_m <= 0:
        parser.error("Initialization distance must be finite and positive")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 11):
        parser.error("--max-seconds must be finite and >11 (10 s warmup plus a 1 s model window)")
    return args


def preflight(args, models):
    if args.out_root.exists():
        raise ValueError(f"Output already exists; choose a new directory: {args.out_root}")
    for protected in [args.data_root, args.checkpoint_root]:
        if args.out_root.is_relative_to(protected):
            raise ValueError(f"Output must be outside input directories: {protected}")
    folders = [
        f for f in sorted((args.data_root / "recordings").glob("*/phones/*/sensorlogger"))
        if (not args.session or f.parts[-4] == args.session)
        and (not args.person or f.parts[-2] == args.person)
    ]
    if not folders:
        raise ValueError("No matching recordings; --data-root must contain recordings/<session>/phones/")
    required = []
    for folder in folders:
        required.extend([folder / "Gyroscope.csv", folder / "TotalAcceleration.csv",
                         args.data_root / "recordings" / folder.parts[-4] / "reference" /
                         folder.parts[-2] / "trajectory-gt.csv"])
    if "ronin" in models:
        required.extend(args.checkpoint_root / "Ronin_o2" / name
                        for name in ["checkpoint_38.pt", "config.json"])
    if "tlio" in models:
        required.extend(args.checkpoint_root / "TLIO_o2" / name
                        for name in ["checkpoint_best.pt", "parameters.json"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing required inputs:\n" + "\n".join(missing))
    return [{"session": f.parts[-4], "person": f.parts[-2]} for f in folders]


def alignment_directory(distance_m):
    return f"alignment-{distance_m:g}m-continuous"


def commands(args, models):
    def command(script, *options):
        return [sys.executable, "-B", "-u", str(ROOT / "scripts" / script),
                *map(str, options)]

    filters = []
    for option, value in [("--session", args.session), ("--person", args.person),
                          ("--max-seconds", args.max_seconds)]:
        if value is not None:
            filters.extend([option, str(value)])
    if args.cpu:
        filters.append("--cpu")
    steps = []
    for index, model in enumerate(models):
        root = args.out_root / model
        checkpoint = args.checkpoint_root if model == "ronin" else args.checkpoint_root / "TLIO_o2"
        inference = command("ronin_pedroam_6d.py" if model == "ronin" else "tlio_pedroam.py",
                            "--data-root", args.data_root, "--checkpoint-root", checkpoint,
                            "--out-root", root / "inference",
                            "--initialization-distance-m", args.initialization_distance_m, *filters)
        if model == "tlio":
            inference.extend(["--workers", str(args.workers)])
        steps.append((f"{model}-inference", inference))
        # Both baselines consume the same selected reference files; audit once.
        if index == 0:
            steps.append(("reference-audit", command(
                "audit_reference.py", "--data-root", args.data_root,
                "--result-root", root / "inference", "--out-root", args.out_root / "reference-audit")))
        # Both models export initial-aligned trajectories and comparison plots
        # during inference. This stage reuses their transforms for reports.
        steps.append((f"{model}-alignment", command(
            "evaluate_initial_alignment.py", "--result-root", root / "inference",
            "--out-root", root / alignment_directory(args.initialization_distance_m),
            "--initialization-distance-m", args.initialization_distance_m,
            "--label", LABELS[model], "--protocol", "continuous")))
    return steps


def merge_tables(out, models, source_name, target_name, alignment=ALIGNMENT):
    """Collect model tables for any number of selected recordings."""
    rows, fields = [], None
    for model in models:
        with (out / model / alignment / source_name).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if fields is None:
                fields = reader.fieldnames
            elif fields != reader.fieldnames:
                raise ValueError("Model result table schemas differ")
            rows.extend({"baseline": model, **row} for row in reader)
    with (out / target_name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["baseline", *fields])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    models = ["ronin", "tlio"] if args.model == "both" else [args.model]
    try:
        records = preflight(args, models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    steps = commands(args, models)
    print(f"Selected {len(records)} recordings; models: {', '.join(models)}", flush=True)
    print(f"Results: {args.out_root}", flush=True)
    if args.dry_run:
        for stage, command in steps:
            print(f"\n[{stage}]\n{shlex.join(command)}")
        return

    # Fail before creating results if the current Python environment is incomplete.
    import torch
    import numpy, pandas, scipy, ahrs, numba, matplotlib  # noqa: F401
    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; rerun with --cpu.")
    args.out_root.mkdir(parents=True, exist_ok=False)
    (args.out_root / "logs").mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               NUMBA_CACHE_DIR=str(args.out_root / ".cache/numba"),
               MPLCONFIGDIR=str(args.out_root / ".cache/matplotlib"))
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "recordings": records, "protocol": "continuous", "steps": [],
    }

    def save_manifest():
        (args.out_root / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    try:
        for stage, command in steps:
            entry = {"stage": stage, "command": command, "status": "running"}
            manifest["steps"].append(entry)
            save_manifest()
            print(f"\n[{stage}]\n{shlex.join(command)}", flush=True)
            with (args.out_root / "logs" / f"{stage}.log").open("w") as log:
                with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
            entry.update(status="completed" if returncode == 0 else "failed", returncode=returncode)
            save_manifest()
            if returncode:
                raise RuntimeError(f"{stage} failed; see logs/{stage}.log. Partial results have been retained.")
        alignment = alignment_directory(args.initialization_distance_m)
        merge_tables(args.out_root, models, "aggregate.csv", "summary.csv", alignment)
        merge_tables(args.out_root, models, "per_sequence.csv", "per_sequence.csv", alignment)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
    print(f"\nCompleted. Aggregate metrics: {args.out_root / 'summary.csv'}")
    print(f"Per-recording metrics: {args.out_root / 'per_sequence.csv'}")


if __name__ == "__main__":
    main()
