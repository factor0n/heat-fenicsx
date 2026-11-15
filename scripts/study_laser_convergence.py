"""
Mesh and time-step independence of the laser case, plus an MPI test.
Runs laser.py as a subprocess and collects the JSON summaries into a CSV.
"""
import csv
import json
import os
import subprocess
import sys

from common import RESULTS

PY = sys.executable
MPIRUN = os.path.join(os.path.dirname(PY), "mpirun")
HERE = os.path.dirname(os.path.abspath(__file__))

runs = [
    # (tag, nprocs, args)
    ("_nx100", 1, ["--nx", "100"]),
    ("_nx200", 1, ["--nx", "200"]),
    ("_nx400", 1, ["--nx", "400"]),
    ("_dt0.1", 1, ["--nx", "200", "--dt", "0.1"]),
    ("_dt0.025", 1, ["--nx", "200", "--dt", "0.025"]),
    ("_np1_cg", 1, ["--nx", "400", "--solver", "cg"]),
    ("_np2_cg", 2, ["--nx", "400", "--solver", "cg"]),
    ("_np2_lu", 2, ["--nx", "400", "--solver", "lu"]),
]

rows = []
for tag, nprocs, extra in runs:
    cmd = [PY, "laser.py", "--no-output", "--tag", tag] + extra
    if nprocs > 1:
        cmd = [MPIRUN, "-n", str(nprocs)] + cmd
    print(" ".join(cmd[-8:]), flush=True)
    subprocess.run(cmd, cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    with open(RESULTS / f"laser_summary{tag}.json") as fh:
        s = json.load(fh)
    rows.append(dict(tag=tag.strip("_"), nx=s["nx"], dt=s["dt"], ndofs=s["ndofs"], np=s["mpi_size"],
                     solver=s["solver"], Tmax=s["Tmax_peak"], probe0=s["probe_peaks"][0],
                     probe5=s["probe_peaks"][1], probe10=s["probe_peaks"][2],
                     balance_err=s["energy_balance_rel_err"], ksp_its=s["mean_ksp_its"],
                     wall_s=s["wall_time_s"]))
    print(f"   Tmax={s['Tmax_peak']:.2f}  sonde0={s['probe_peaks'][0]:.2f}  wall={s['wall_time_s']:.1f}s")

with open(RESULTS / "laser_convergence.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("-> results/laser_convergence.csv")
