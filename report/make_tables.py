"""
Builds report/generated.tex (tables + number macros) from the CSV/JSON files in results/.
No number in the report is copied by hand.
"""
import csv
import re
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
out = []


def sci(x, d=2):
    m, e = f"{float(x):.{d}e}".split("e")
    return rf"$\num{{{m}}}\cdot 10^{{{int(e)}}}$"


def rate(x):
    return "--" if x in ("nan", "") else rf"\num{{{float(x):.2f}}}"


def macro(name, val):
    if re.fullmatch(r"[-+]?[0-9.]+", str(val)):
        val = rf"\num{{{val}}}"
    out.append(rf"\newcommand{{\{name}}}{{{val}}}")


# ---------------------------------------------------------------- convergence in space
rows = list(csv.DictReader(open(R / "convergence_space.csv")))
out.append(r"\newcommand{\TabSpace}{%")
out.append(r"\begin{tabular}{c r r c c c c}\toprule")
out.append(r"Élément & $N$ & ddl & $\|e\|_{L^2}$ & ordre & $|e|_{H^1}$ & ordre \\ \midrule")
for i, r in enumerate(rows):
    lab = rf"$\mathbb{{P}}_{r['degree']}$" if r["N"] == "4" else ""
    out.append(rf"{lab} & {r['N']} & {int(r['ndofs']):,} & {sci(r['err_L2'])} & {rate(r['rate_L2'])} & "
               rf"{sci(r['err_H1'])} & {rate(r['rate_H1'])} \\".replace(",", "\\,"))
    if r["N"] == "64" and i < len(rows) - 1:
        out.append(r"\midrule")
out.append(r"\bottomrule\end{tabular}}")

# ---------------------------------------------------------------- convergence in time
rows = list(csv.DictReader(open(R / "convergence_time.csv")))
be = [r for r in rows if float(r["theta"]) == 1.0]
cn = [r for r in rows if float(r["theta"]) == 0.5]
out.append(r"\newcommand{\TabTime}{%")
out.append(r"\begin{tabular}{c r c c c c}\toprule")
out.append(r" & & \multicolumn{2}{c}{Euler implicite ($\theta=1$)} & \multicolumn{2}{c}{Crank--Nicolson ($\theta=\frac12$)} \\")
out.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
out.append(r"$\Delta t$ & pas & $\|e(T)\|_{L^2}$ & ordre & $\|e(T)\|_{L^2}$ & ordre \\ \midrule")
for a, b in zip(be, cn):
    out.append(rf"$\num{{{float(a['dt']):g}}}$ & {a['n_steps']} & {sci(a['err_L2'])} & {rate(a['rate_L2'])} & "
               rf"{sci(b['err_L2'])} & {rate(b['rate_L2'])} \\")
out.append(r"\bottomrule\end{tabular}}")

# ---------------------------------------------------------------- stability
s = json.load(open(R / "stability.json"))
macro("LamMin", f"{s['lambda_min']:.3f}")
macro("LamMinExact", f"{s['lambda_min_exact']:.3f}")
macro("LamMax", f"{s['lambda_max']:.1f}")
macro("DtC", sci(s["dt_c"], 3).strip("$"))
macro("RatioStable", f"{s['cfl']['0.95']['ratio']:.3f}")
macro("RatioUnstable", f"{s['cfl']['1.05']['ratio']:.3f}")
macro("NormStable", f"{s['cfl']['0.95']['final_norm']:.3f}")
macro("NormUnstable", sci(s["cfl"]["1.05"]["final_norm"], 1).strip("$"))
mm = s["rough_data"]["min_max"]
macro("CNmin", f"{mm['Crank-Nicolson'][0]:.3f}")
macro("CNmax", f"{mm['Crank-Nicolson'][1]:.3f}")
macro("BEmin", f"{mm['Euler implicite'][0]:.3f}")
macro("BEmax", f"{mm['Euler implicite'][1]:.3f}")
macro("REFmax", f"{mm['Référence'][1]:.3f}")

# ---------------------------------------------------------------- laser
L = json.load(open(R / "laser_summary.json"))
macro("LaserTmax", f"{L['Tmax_peak']:.1f}")
macro("LaserProbeA", f"{L['probe_peaks'][0]:.1f}")
macro("LaserProbeB", f"{L['probe_peaks'][1]:.1f}")
macro("LaserProbeC", f"{L['probe_peaks'][2]:.1f}")
macro("LaserEin", f"{L['energy_in_J']:.2f}")
macro("LaserEout", f"{L['energy_out_J']:.1f}")
macro("LaserEstored", f"{L['energy_stored_J']:.1f}")
macro("LaserLossPct", f"{100 * L['energy_out_J'] / L['energy_in_J']:.1f}")
macro("LaserBalance", sci(L["energy_balance_rel_err"], 1).strip("$"))
macro("LaserNdofs", f"{L['ndofs']:,}".replace(",", "\\,"))
macro("LaserWall", f"{L['wall_time_s']:.0f}")

rows = list(csv.DictReader(open(R / "laser_convergence.csv")))
by = {r["tag"]: r for r in rows}
ref = float(by["nx400"]["Tmax"])
out.append(r"\newcommand{\TabLaserConv}{%")
out.append(r"\begin{tabular}{l r r r c c c}\toprule")
out.append(r"Cas & $h$ (mm) & $\Delta t$ (s) & ddl & $\max T$ (°C) & écart / réf. & $T_{\text{sonde }0}$ (°C) \\ \midrule")
for tag in ["nx100", "nx200", "nx400", "dt0.1", "nx200", "dt0.025"]:
    r = by[tag]
    if tag == "dt0.1":
        out.append(r"\midrule")
    hmm = 100.0 / int(r["nx"])
    d = float(r["Tmax"]) - ref
    lab = (rf"maillage ${int(r['nx'])}\times{int(r['nx'])//2}$" if tag.startswith("nx")
           else rf"pas $\Delta t = \num{{{float(r['dt']):g}}}$")
    out.append(rf"{lab} & \num{{{hmm:g}}} & \num{{{float(r['dt']):g}}} & "
               rf"{int(r['ndofs']):,} & \num{{{float(r['Tmax']):.2f}}} & \num{{{d:+.2f}}} & \num{{{float(r['probe0']):.2f}}} \\".replace(",", "\\,"))
out.append(r"\bottomrule\end{tabular}}")

out.append(r"\newcommand{\TabMPI}{%")
out.append(r"\begin{tabular}{c l c r c c}\toprule")
out.append(r"Processus MPI & Solveur & it. KSP / pas & temps (s) & accélération & $\max T$ (°C) \\ \midrule")
base = {"lu": float(by["nx400"]["wall_s"]), "cg": float(by["np1_cg"]["wall_s"])}
for tag in ["nx400", "np2_lu", "np1_cg", "np2_cg"]:
    r = by[tag]
    sol = "LU (MUMPS)" if r["solver"] == "lu" else "CG + BoomerAMG"
    sp = base[r["solver"]] / float(r["wall_s"])
    out.append(rf"{r['np']} & {sol} & \num{{{float(r['ksp_its']):.1f}}} & \num{{{float(r['wall_s']):.1f}}} & \num{{{sp:.2f}}} & "
               rf"\num{{{float(r['Tmax']):.6f}}} \\")
out.append(r"\bottomrule\end{tabular}}")
macro("NdofsMPI", f"{int(by['nx400']['ndofs']):,}".replace(",", "\\,"))

# ---------------------------------------------------------------- strong scaling (run on the author's PC)
S = json.load(open(R / "scaling_summary.json"))
runs = list(csv.DictReader(open(R / "scaling.csv")))
macro("ScalCPU", S["cpu"].replace("(R)", r"\textsuperscript{\textregistered}").replace("(TM)", r"\texttrademark{}"))
macro("ScalNdofs", f"{S['ndofs']:,}".replace(",", "\\,"))
macro("ScalReps", str(S["reps"]))
macro("ScalTmaxSpread", sci(S["Tmax_max_minus_min"], 1).strip("$"))
procs = sorted(int(k) for k in S["median_wall_s"])
spread = {}
for q in procs:
    w = [float(r["wall_s"]) for r in runs if int(r["np"]) == q]
    spread[q] = 100 * (max(w) - min(w)) / min(w)
macro("ScalMaxSpread", f"{max(spread.values()):.1f}")
for q in procs:
    name = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}.get(q, f"P{q}")
    macro(f"ScalS{name}", f"{S['speedup'][str(q)]:.2f}")
    macro(f"ScalE{name}", f"{100 * S['efficiency'][str(q)]:.0f}")
    macro(f"ScalT{name}", f"{S['median_wall_s'][str(q)]:.1f}")
out.append(r"\newcommand{\TabScaling}{%")
out.append(r"\begin{tabular}{c r c c c c}\toprule")
out.append(r"Processus & temps médian (s) & dispersion & it.\ CG / pas & accélération $S(p)$ & efficacité $E(p)$ \\ \midrule")
for q in procs:
    its = float([r for r in runs if int(r["np"]) == q][0]["ksp_its"])
    out.append(rf"{q} & \num{{{S['median_wall_s'][str(q)]:.1f}}} & \num{{{spread[q]:.1f}}}~\% & "
               rf"\num{{{its:.2f}}} & \num{{{S['speedup'][str(q)]:.2f}}} & \num{{{100 * S['efficiency'][str(q)]:.0f}}}~\% \\")
out.append(r"\bottomrule\end{tabular}}")

(Path(__file__).parent / "generated.tex").write_text("\n".join(out) + "\n")
print("-> report/generated.tex")
