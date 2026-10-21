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
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = "alignment-10m-continuous"
LABELS = {"ronin": "Madgwick", "tlio": "TLIO-EKF"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Dataset directory containing recordings/")
    parser.add_argument("--out-root", type=Path,
                        help="Results base directory (default: <data-root>/results)")
    parser.add_argument("--test-name",
                        help="Experiment directory name (default: unique eqnio-baselines timestamp)")
    parser.add_argument("--work-root", type=Path,
                        help="Fresh intermediate run directory (default: repository outputs/baseline-work/<test-name>/<timestamp>)")
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
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if args.test_name is None:
        args.test_name = f"eqnio-baselines-{run_id}"
    if (not args.test_name or args.test_name in {".", ".."}
            or any(char in args.test_name for char in ["/", "\\", "\x00"])):
        parser.error("--test-name must be a single nonempty directory name")
    if args.out_root is None:
        args.out_root = args.data_root.expanduser() / "results"
    if args.work_root is None:
        args.work_root = ROOT / "outputs/baseline-work" / args.test_name / run_id
    for field in ["data_root", "out_root", "work_root", "checkpoint_root"]:
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not math.isfinite(args.initialization_distance_m) or args.initialization_distance_m <= 0:
        parser.error("Initialization distance must be finite and positive")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 11):
        parser.error("--max-seconds must be finite and >11 (10 s warmup plus a 1 s model window)")
    return args


def preflight(args, models):
    if args.work_root.exists():
        raise ValueError(f"Work directory already exists; choose a new --test-name or --work-root: {args.work_root}")
    if args.out_root.exists() and not args.out_root.is_dir():
        raise ValueError(f"Results base is not a directory: {args.out_root}")
    allowed_results = (args.data_root / "results").resolve()
    protected_data = [(args.data_root / name).resolve() for name in ["recordings", "rosbags", "environment"]]
    if args.out_root == args.data_root or any(args.out_root.is_relative_to(path) for path in protected_data):
        raise ValueError("Results must not overlap PedROAM input data")
    if args.out_root.is_relative_to(args.data_root) and not args.out_root.is_relative_to(allowed_results):
        raise ValueError("Within PedROAM, results must be under <data-root>/results, never recordings or other inputs")
    if args.out_root.is_relative_to(args.checkpoint_root):
        raise ValueError("Results must be outside the checkpoint directory")
    for protected in [args.data_root, args.checkpoint_root]:
        if args.work_root.is_relative_to(protected):
            raise ValueError(f"Work directory must be outside input directories: {protected}")
    if args.work_root.is_relative_to(args.out_root) or args.out_root.is_relative_to(args.work_root):
        raise ValueError("Work and results directories must not contain each other")
    folders = [
        f for f in sorted((args.data_root / "recordings").glob("*/phones/*/sensorlogger"))
        if (not args.session or f.parts[-4] == args.session)
        and (not args.person or f.parts[-2] == args.person)
    ]
    if not folders:
        raise ValueError("No matching recordings; --data-root must contain recordings/<session>/phones/")
    for folder in folders:
        record = {"session": folder.parts[-4], "person": folder.parts[-2]}
        target = result_directory(args, record)
        resolved = target.resolve()
        if not resolved.is_relative_to(args.out_root) or resolved.is_relative_to(args.checkpoint_root):
            raise ValueError(f"Result path escapes its base or overlaps checkpoints: {target}")
        if resolved.is_relative_to(args.data_root) and not resolved.is_relative_to(allowed_results):
            raise ValueError(f"Result path overlaps input data: {target}")
        if any(resolved.is_relative_to(path) for path in protected_data):
            raise ValueError(f"Result path overlaps input data: {target}")
        if target.exists() or target.is_symlink():
            raise ValueError(f"Experiment already exists; choose a new --test-name: {target}")
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


def result_directory(args, record):
    return args.out_root / record["session"] / record["person"] / args.test_name


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
        root = args.work_root / model
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
                "--result-root", root / "inference", "--out-root", args.work_root / "reference-audit")))
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


def copy_record_table(source, target, record):
    """Keep only the selected session/person when archiving batch tables."""
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if all(row[key] == value for key, value in record.items())]
    with target.open("w", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def publish_results(args, models, records, manifest):
    """Archive self-contained per-record experiments without overwriting results.

    Assemble all bundles in the work directory first. Only completed inference
    and reports are published. A failed copy retains a publishing status rather
    than presenting partial output as a completed experiment.
    """
    alignment = alignment_directory(args.initialization_distance_m)
    bundles = []
    for record in records:
        destination = result_directory(args, record)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Experiment already exists: {destination}")
        bundle = args.work_root / "archive" / record["session"] / record["person"] / args.test_name
        bundle.mkdir(parents=True, exist_ok=False)
        for model in models:
            inference_root = args.work_root / model / "inference"
            inference_run = json.loads((inference_root / "run.json").read_text())
            subdir = inference_run.get("trajectory_subdir", f"eqnio-ronin-o2-{inference_run['method']}")
            common = inference_root / record["session"] / record["person"]
            shutil.copytree(common / subdir, bundle / model)
            for path in common.iterdir():
                if path.is_file():
                    shutil.copy2(path, bundle / model / path.name)
            shutil.copy2(inference_root / "run.json", bundle / model / "run.json")
            report = args.work_root / model / alignment / LABELS[model] / record["session"] / record["person"]
            shutil.copytree(report, bundle / model / "evaluation")
            shutil.copy2(args.work_root / model / alignment / "run.json", bundle / model / "evaluation/run.json")
            summary_path = bundle / model / "evaluation/summary.json"
            summary = json.loads(summary_path.read_text())
            summary["source_original"] = summary["source"]
            summary["source"] = str(destination / model)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        audit_root = args.work_root / "reference-audit"
        shutil.copytree(audit_root / record["session"] / record["person"], bundle / "reference-audit")
        audit = json.loads((audit_root / "audit.json").read_text())
        key = f"{record['session']}/{record['person']}"
        audit["sources"] = {key: audit["sources"][key]}
        audit["scope"] = "selected recording; raw matching metadata inspected"
        (bundle / "reference-audit/audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        for name in ["per_sequence.csv", "discontinuities.csv"]:
            copy_record_table(audit_root / name, bundle / "reference-audit" / name, record)
        copy_record_table(args.work_root / "per_sequence.csv", bundle / "summary.csv", record)
        bundle_manifest = {
            **manifest, "status": "publishing", "recordings": [record],
            "result_directories": [str(destination)], "published_results": [],
            "test_name": args.test_name, "result_directory": str(destination),
            "work_directory": str(args.work_root),
            "layout": "results/<session>/<person>/<test-name>/<model>/",
            "logs": str(args.work_root / "logs"),
            "summary": "summary.csv contains this recording's initial-alignment metrics for each model",
        }
        (bundle / "run.json").write_text(json.dumps(bundle_manifest, indent=2) + "\n")
        bundles.append((bundle, destination, bundle_manifest))
    for bundle, destination, bundle_manifest in bundles:
        destination.mkdir(parents=True, exist_ok=False)  # Reserve without overwriting a concurrent run.
        shutil.copy2(bundle / "run.json", destination / "run.json")
        for path in bundle.iterdir():
            if path.name == "run.json":
                continue
            if path.is_dir():
                shutil.copytree(path, destination / path.name)
            else:
                shutil.copy2(path, destination / path.name)
        bundle_manifest.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                               published_results=[str(destination)])
        (destination / "run.json").write_text(json.dumps(bundle_manifest, indent=2) + "\n")
        manifest["published_results"].append(str(destination))
        print(f"Archived: {destination}", flush=True)


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
    print(f"Work directory: {args.work_root}", flush=True)
    for record in records:
        print(f"Experiment: {result_directory(args, record)}", flush=True)
    if args.dry_run:
        for stage, command in steps:
            print(f"\n[{stage}]\n{shlex.join(command)}")
        return

    # Fail before creating results if the current Python environment is incomplete.
    import torch
    import numpy, pandas, scipy, ahrs, numba, matplotlib  # noqa: F401
    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; rerun with --cpu.")
    args.work_root.mkdir(parents=True, exist_ok=False)
    (args.work_root / "logs").mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               NUMBA_CACHE_DIR=str(args.work_root / ".cache/numba"),
               MPLCONFIGDIR=str(args.work_root / ".cache/matplotlib"))
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "recordings": records, "protocol": "continuous", "steps": [], "published_results": [],
        "result_directories": [str(result_directory(args, record)) for record in records],
    }

    def save_manifest():
        (args.work_root / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    try:
        for stage, command in steps:
            entry = {"stage": stage, "command": command, "status": "running"}
            manifest["steps"].append(entry)
            save_manifest()
            print(f"\n[{stage}]\n{shlex.join(command)}", flush=True)
            with (args.work_root / "logs" / f"{stage}.log").open("w") as log:
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
        merge_tables(args.work_root, models, "aggregate.csv", "summary.csv", alignment)
        merge_tables(args.work_root, models, "per_sequence.csv", "per_sequence.csv", alignment)
        publish_results(args, models, records, manifest)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
    print(f"\nCompleted. Official experiments: {args.out_root}/<session>/<person>/{args.test_name}/")
    print(f"Aggregate metrics: {args.work_root / 'summary.csv'}")
    print(f"Per-recording metrics: {args.work_root / 'per_sequence.csv'}")


if __name__ == "__main__":
    main()
