#!/usr/bin/env python3
"""Audit the reference samples consumed by inference without editing raw data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_initial_alignment import digest, plt, cumulative_distance_index
from reference_quality import reference_continuity, continuous_distance_interval


def audit(result_root, data_root, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((result_root / "run.json").read_text())
    rows, events, source_hashes = [], [], {}
    table = pd.read_csv(result_root / "summary.csv")
    for record in table[table.variant == "o2"].sort_values(["session", "person"]).itertuples():
        key = f"{record.session}/{record.person}"
        source = result_root / key / manifest.get(
            "trajectory_subdir", f"eqnio-ronin-o2-{manifest['method']}")
        ev = pd.read_csv(source / "evaluation.csv")
        provenance = json.loads((source / "summary.json").read_text())
        raw_path = data_root / "recordings" / record.session / "reference" / record.person / "trajectory-gt.csv"
        raw_hash = digest(raw_path)
        assert raw_hash == provenance["reference_sha256"], f"Reference changed: {key}"
        raw = pd.read_csv(raw_path)
        ns = ev.timestamp_ns.to_numpy(np.int64)
        t = (ns - ns[0]) * 1e-9
        ref = ev[["gt_e_m", "gt_n_m"]].to_numpy()
        quality = reference_continuity(t, ref)
        i, j = continuous_distance_interval(t, ref)
        old_j = cumulative_distance_index(ref)
        target = out / key
        target.mkdir(parents=True)
        pd.DataFrame({"timestamp_ns": ns, "elapsed_s": t, **quality}).to_csv(
            target / "reference_quality.csv", index=False)
        raw_ns = np.rint(raw.observation_utc_seconds.to_numpy() * 1e9).astype(np.int64)
        valid = raw.gt_valid.astype(str).str.lower().eq("true").to_numpy()
        for k in np.flatnonzero(quality["gt_discontinuity"]):
            between = (raw_ns > ns[k - 1]) & (raw_ns < ns[k])
            invalid = raw.loc[between & ~valid]
            reasons = sorted(invalid.invalid_reason.dropna().unique().tolist())
            events.append({
                "session": record.session, "person": record.person,
                "left_timestamp_ns": int(ns[k - 1]), "right_timestamp_ns": int(ns[k]),
                "elapsed_s": t[k], "dt_s": quality["gt_dt_s"][k],
                "step_m": quality["gt_step_m"][k], "speed_mps": quality["gt_speed_mps"][k],
                "gap_flag": bool(quality["gt_gap_flag"][k]),
                "speed_flag": bool(quality["gt_speed_flag"][k]),
                "invalid_rows_between": len(invalid), "invalid_reasons": ";".join(reasons),
            })
        # Retain upstream match metadata around the initial window for diagnosis.
        raw.loc[(raw_ns >= ns[0] - 2_000_000_000) &
                (raw_ns <= ns[j] + 5_000_000_000)].to_csv(target / "initial_raw_reference.csv", index=False)
        rows.append({
            "session": record.session, "person": record.person,
            "matched_samples": len(t), "speed_flags": int(quality["gt_speed_flag"].sum()),
            "gap_flags": int(quality["gt_gap_flag"].sum()),
            "max_speed_mps": float(quality["gt_speed_mps"].max()),
            "legacy_end_s": float(t[old_j]),
            "continuous_start_s": float(t[i]), "continuous_end_s": float(t[j]),
            "legacy_gt_net_m": float(np.linalg.norm(ref[old_j] - ref[0])),
            "continuous_gt_net_m": float(np.linalg.norm(ref[j] - ref[i])),
        })
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        plot_end = max(60, t[j] + 10)
        early = t <= plot_end
        axes[0].plot(t[early], ref[early, 0], label="GT east")
        axes[0].plot(t[early], ref[early, 1], label="GT north")
        axes[0].set(ylabel="Position (m)")
        axes[1].plot(t[early], quality["gt_speed_mps"][early], label="GT step / time")
        axes[1].axhline(3., color="red", linestyle=":", label="3 m/s screening threshold")
        axes[1].set(ylabel="Horizontal speed (m/s)", xlabel="Seconds from first matched GT")
        for ax in axes:
            ax.axvline(t[old_j], color="red", linestyle="--", label="Legacy init end")
            ax.axvspan(t[i], t[j], color="green", alpha=.15, label="Continuous 10 m interval")
            ax.set_xlim(0, plot_end)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle(key)
        fig.savefig(target / "initial_reference.png", dpi=150)
        plt.close(fig)
        assert digest(raw_path) == raw_hash
        source_hashes[key] = {"reference_sha256": raw_hash,
                              "source_evaluation_sha256": digest(source / "evaluation.csv")}
    pd.DataFrame(rows).to_csv(out / "per_sequence.csv", index=False)
    pd.DataFrame(events).to_csv(out / "discontinuities.csv", index=False)
    (out / "audit.json").write_text(json.dumps({
        "result_root": str(result_root.resolve()), "data_root": str(data_root.resolve()),
        "script_sha256": digest(Path(__file__)),
        "quality_script_sha256": digest(Path(__file__).with_name("reference_quality.py")),
        "thresholds": {"max_gap_s": 1.5, "max_speed_mps": 3., "settling_s": 10.},
        "scope": "all GT samples used by saved inference; raw matching metadata inspected",
        "limitations": "Flags identify suspect transitions, not proven sensor faults. No raw GT repair.",
        "sources": source_hashes, "references_unchanged": True,
    }, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("outputs/ronin-current/inference"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/maxson/data/PedROAM"))
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.result_root, args.data_root, args.out_root)
