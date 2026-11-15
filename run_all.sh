#!/usr/bin/env bash
# Reproduces all results and figures of the report (~10 min on 2 cores).
set -euo pipefail
cd "$(dirname "$0")/scripts"

python study_space.py               # convergence in space  (P1, P2, P3)
python study_time.py                # convergence in time   (backward Euler, Crank-Nicolson)
python study_stability.py           # forward Euler CFL limit + Crank-Nicolson oscillations
python laser.py                     # application: moving laser source (figures + XDMF)
python study_laser_convergence.py   # mesh / time-step independence + MPI

echo "Résultats dans results/ ; ouvrir results/laser.xdmf dans ParaView."
