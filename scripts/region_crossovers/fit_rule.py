#!/usr/bin/env python3
"""Fit and check the walk rule of MeshInspector/MeshLib#7134 (chooseWalk) on region_crossovers results.

For a set of `count` elements touching `blocks` non-empty 64-bit blocks in a mesh of `size` elements:
    serial walk    iff  count * dense * M < size  and  blocks * scattered * M < size
    else parallel  iff  count * par <= size        (only the functions with a parMs column)
    else scan
where M = 1 for fixed fractions, or M = the machine's thread count for the thread-scaled rule
(the parallel walk beats the serial one at a set size about proportional to the thread count).
blocks = count / setFill. The limits are fitted per function by a grid search over all machines and layouts:
first minimize the total loss beyond noise, sum of max(0, log(chosen / fastest / NOISE)) with ABS_MS slack
for the timer on tiny sets, then the mean log ratio.
Usage: fit_rule.py <dir with <machine>/results.csv and <machine>/info.txt> [<dir of another run>...]
Several runs are merged by the minimum time of each implementation per point."""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SIZES = {"dragon-1M.stl": {"faces": 1025766, "verts": 512880, "edges": 1538654},
         "dragon-250k.stl": {"faces": 244026, "verts": 122005, "edges": 366039}}
KIND = lambda fn: "verts" if fn.endswith("(verts)") else "edges" if fn.endswith("(edges)") else "faces"
ABS_MS = 0.002  # differences below 2 us are timer noise
NOISE = 1.1
# current limits in the PR (dense, scattered, parallel); None = no parallel tier
CURRENT = {"getIncidentEdges(faces)": (32, 128, 8), "getIncidentVerts(faces)": (16, 128, 8),
           "getInnerVerts(faces)": (128, 256, None), "getRegionBoundaryVerts": (16, 256, None),
           "getRegionInnerVerts": (128, 512, None), "getInnerEdges(faces)": (64, 256, 8),
           "findRegionBoundaryUndirectedEdgesInsideMesh": (64, 256, 8), "findAllLeftBdEdges": (32, 256, 4),
           "getIncidentFaces(verts)": (64, 512, 16), "getInnerFaces(verts)": (128, 2048, None),
           "getInnerEdges(verts)": (128, 1024, 32), "getIncidentEdges(edges)": (32, 256, 8)}
FIXED = [2 ** k for k in range(1, 16)]
SCALED = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
PARS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]

points = {}
threads = {}
for root in sys.argv[1:]:
    for m in sorted(Path(root).iterdir()):
        if not (m / "results.csv").exists():
            continue
        threads[m.name] = int(re.search(r"(\d+) threads", (m / "info.txt").read_text()).group(1))
        for r in csv.DictReader(open(m / "results.csv")):
            t = [float(r["scanMs"]), float(r["walkMs"]), float(r["parMs"]) if r.get("parMs") else np.inf]
            key = (m.name, r["mesh"], r["layout"], r["function"], r["fraction"])
            if key in points:
                t = [min(a, b) for a, b in zip(t, points[key][1])]
            points[key] = (r, t)

data = defaultdict(lambda: defaultdict(list))
for (machine, mesh, layout, fn, fraction), (r, t) in points.items():
    d = data[fn]
    count, fill = int(r["setSize"]), float(r["setFill"])
    d["machine"].append(machine)
    d["threads"].append(threads[machine])
    d["where"].append(f"{mesh[7:-4]} {layout} 1/{fraction}")
    d["size"].append(SIZES[mesh][KIND(fn)])
    d["count"].append(count)
    d["blocks"].append(count / fill if fill else 0)
    d["t"].append(t)
data = {fn: {k: np.array(v) for k, v in d.items()} for fn, d in data.items()}
machines = sorted(threads, key=lambda m: (threads[m], m))




def choose(d, lim, scaled):
    dense, scattered, par = lim
    mul = d["threads"] if scaled else 1
    serial = (d["count"] * dense * mul < d["size"]) & (d["blocks"] * scattered * mul < d["size"])
    if not par:
        return np.where(serial, 1, 0)
    parallel = d["count"] * par <= d["size"]
    return np.where(serial, 1, np.where(parallel, 2, 0))


def ratios(d, choice):
    chosen = d["t"][np.arange(len(choice)), choice]
    best = d["t"].min(axis=1)
    return chosen / best, chosen / (best + ABS_MS)


def fit(d, scaled):
    hasPar = np.isfinite(d["t"][:, 2]).all()
    grid = SCALED if scaled else FIXED
    best = None
    for dense in grid:
        for scattered in grid:
            if scattered < dense:
                continue
            for par in (p for p in PARS if scaled or p <= dense) if hasPar else [None]:
                lim = (dense, scattered, par)
                _, adj = ratios(d, choose(d, lim, scaled))
                logs = np.log(adj)
                key = (np.maximum(0, logs - np.log(NOISE)).sum(), logs.mean())
                if best is None or key < best[0]:
                    best = (key, lim)
    return best[1]


fmt = lambda x: "/".join(f"{v:g}" if v else "-" for v in x)


def report(title, limits, scaled):
    print(f"\n{title}\nworst chosen/fastest per machine (threads), with {ABS_MS * 1000:g} us slack; "
          f"last column = rows losing more than {NOISE}x / all rows")
    print(f"{'function':44} {'limits':>16} " + " ".join(f"{m[:11] + '(' + str(threads[m]) + ')':>15}" for m in machines) + "    lost")
    for fn, d in data.items():
        lim = limits[fn]
        _, adj = ratios(d, choose(d, lim, scaled))
        cells = [f"{adj[d['machine'] == m].max():.2f}" for m in machines]
        print(f"{fn:44} {fmt(lim):>16} " + " ".join(f"{c:>15}" for c in cells) + f"  {(adj > NOISE).sum():3}/{len(adj)}")


def worstRows(limits, scaled, n=3):
    print("\nworst rows: chosen/fastest, times scan/serial/parallel ms")
    names = ["scan", "serial", "parallel"]
    for fn, d in data.items():
        choice = choose(d, limits[fn], scaled)
        _, adj = ratios(d, choice)
        for i in np.argsort(-adj)[:n]:
            t = "/".join(f"{x:.3f}" for x in d["t"][i] if np.isfinite(x))
            print(f"  {fn:44} {adj[i]:5.2f} {names[choice[i]]:8} {d['machine'][i]:15} {d['where'][i]:34} {t}")


report("Current limits in the PR (dense/scattered/parallel)", {fn: CURRENT[fn] for fn in data}, False)
for scaled, title in [(False, "A. fixed fractions, the PR's form (dense/scattered/parallel)"),
                      (True, "B. serial limits times the thread count T (dense*T/scattered*T/parallel)")]:
    limits = {fn: fit(d, scaled) for fn, d in data.items()}
    report(title, limits, scaled)
    worstRows(limits, scaled)
