"""
Strong scaling study of the laser case: same problem, more and more MPI processes.

For each process count we run laser.py several times and keep the median wall time.
Outputs:
    results/scaling.csv               one line per run
    results/scaling_summary.json      median times, speedup, efficiency, machine info
    results/figures/scaling.png       speedup and parallel efficiency plots

Usage (from the scripts/ folder):
    python study_scaling.py                          # default: nx=600, CG, np = 1 2 4 8, 3 repeats
    python study_scaling.py --nx 400 --np 1 2 4 --reps 2
"""
import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGS, RESULTS

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=600, help="cells along x (ny = nx/2)")
p.add_argument("--solver", default="cg", choices=["lu", "cg"])
p.add_argument("--np", type=int, nargs="+", default=[1, 2, 4, 8], help="process counts to test")
p.add_argument("--reps", type=int, default=3, help="repeats per process count")
args = p.parse_args()

PY = sys.executable
MPIRUN = os.path.join(os.path.dirname(PY), "mpirun")
HERE = os.path.dirname(os.path.abspath(__file__))

# One thread per MPI process: stop BLAS / OpenMP from using extra threads,
# otherwise the processes fight for the same cores and timings are wrong.
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def cpu_name():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


rows = []
for nprocs in args.np:
    for rep in range(1, args.reps + 1):
        tag = f"_scal_np{nprocs}_r{rep}"
        cmd = [PY, "laser.py", "--no-output", "--tag", tag, "--nx", str(args.nx), "--solver", args.solver]
        # Pin each process to one core so runs are repeatable
        cmd = [MPIRUN, "-n", str(nprocs), "-bind-to", "core"] + cmd
        print(f"np={nprocs}  repeat {rep}/{args.reps} ...", flush=True)
        subprocess.run(cmd, cwd=HERE, check=True, stdout=subprocess.DEVNULL, env=env)
        summary_file = RESULTS / f"laser_summary{tag}.json"
        with open(summary_file) as fh:
            s = json.load(fh)
        summary_file.unlink()  # keep results/ clean: everything goes into scaling.csv
        rows.append(dict(np=nprocs, rep=rep, nx=s["nx"], ndofs=s["ndofs"], solver=s["solver"],
                         wall_s=s["wall_time_s"], ksp_its=s["mean_ksp_its"], Tmax=s["Tmax_peak"]))
        print(f"   wall = {s['wall_time_s']:.1f} s   Tmax = {s['Tmax_peak']:.6f}", flush=True)

with open(RESULTS / "scaling.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Median time per process count, then speedup S(p) = T(1) / T(p) and efficiency E(p) = S(p) / p
procs = sorted(set(r["np"] for r in rows))
median = {q: statistics.median(r["wall_s"] for r in rows if r["np"] == q) for q in procs}
t_ref = median[procs[0]] * procs[0]  # reference: smallest process count (normally 1)
speedup = {q: t_ref / median[q] for q in procs}
eff = {q: speedup[q] / q for q in procs}
tmax_spread = max(r["Tmax"] for r in rows) - min(r["Tmax"] for r in rows)

summary = dict(cpu=cpu_name(), logical_cpus=os.cpu_count(), nx=args.nx, solver=args.solver,
               ndofs=rows[0]["ndofs"], reps=args.reps,
               median_wall_s=median, speedup=speedup, efficiency=eff,
               Tmax_max_minus_min=tmax_spread)
with open(RESULTS / "scaling_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print("\n np   median time (s)   speedup   efficiency")
for q in procs:
    print(f"{q:3d}   {median[q]:14.1f}   {speedup[q]:7.2f}   {eff[q]:9.2f}")
print(f"Tmax spread over all runs: {tmax_spread:.2e} degC (should be ~1e-8 or smaller)")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.plot(procs, [speedup[q] for q in procs], "o-", label="mesuré")
ax.plot(procs, procs, "k--", lw=1, label="idéal")
ax.set_xlabel("processus MPI")
ax.set_ylabel("accélération $S(p) = T_1 / T_p$")
ax.set_xticks(procs)
ax.grid(alpha=0.3)
ax.legend()
ax = axes[1]
ax.plot(procs, [eff[q] for q in procs], "o-")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_ylim(0, 1.1)
ax.set_xlabel("processus MPI")
ax.set_ylabel("efficacité $E(p) = S(p) / p$")
ax.set_xticks(procs)
ax.grid(alpha=0.3)
fig.suptitle(f"Scalabilité forte — {rows[0]['ndofs']:,} ddl, {args.solver.upper()}, {summary['cpu']}".replace(",", " "),
             fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "scaling.png", dpi=160)
print("-> results/scaling.csv, results/scaling_summary.json, results/figures/scaling.png")
