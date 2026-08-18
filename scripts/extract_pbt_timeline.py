#!/usr/bin/env python3
"""Extract PBT hyperparameter timeline from Ray result.json files (read-only forensics)."""
import json, glob, os, sys

base = sys.argv[1] if len(sys.argv) > 1 else "training_output/v31_500k_20260818_0839/adan_pbt_training"
out_path = sys.argv[2] if len(sys.argv) > 2 else None

trials = sorted(glob.glob(os.path.join(base, "ADAN_PBT_Worker_*")))
lines = []
for t in trials:
    rj = os.path.join(t, "result.json")
    if not os.path.exists(rj):
        continue
    name = os.path.basename(t)
    lines.append("=== " + name[:100] + " ===")
    # checkpoints present in trial dir
    ckpts = sorted(glob.glob(os.path.join(t, "checkpoint_*")))
    lines.append(f"checkpoints_on_disk: {len(ckpts)} -> {[os.path.basename(c) for c in ckpts]}")
    n = 0
    for line in open(rj):
        try:
            r = json.loads(line)
        except Exception:
            continue
        c = r.get("config", {})
        n += 1
        lines.append(
            "iter={it} ts={ts:.0f}s ent={ent} gamma={g} lr={lr} sl={sl} tp={tp}".format(
                it=r.get("training_iteration"),
                ts=float(r.get("time_total_s", 0) or 0),
                ent=c.get("ent_coef"),
                g=c.get("gamma"),
                lr=c.get("learning_rate"),
                sl=c.get("sl_pct"),
                tp=c.get("tp_pct"),
            )
        )
    lines.append(f"total_result_lines={n}")
    lines.append("")

text = "\n".join(lines)
if out_path:
    with open(out_path, "w") as f:
        f.write(text)
print(text)
