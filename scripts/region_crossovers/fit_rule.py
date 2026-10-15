#!/usr/bin/env python3
"""Fit and check the walk rule of MeshInspector/MeshLib#7134 (chooseWalk) on region_crossovers results.

For a set of `count` elements touching `blocks` non-empty 64-bit blocks in a mesh of `size` elements:
    serial walk    iff  count * dense < size  and  blocks * scattered < size
    else parallel  iff  count * par <= size        (only the functions with a parMs column)
    else scan
blocks = count / setFill. All three limits are fitted per function by a grid search over all machines and layouts:
first minimize the worst ratio of the chosen implementation to the fastest one on any row (with ABS_MS slack for
timer noise on tiny sets), then the mean log ratio.
Usage: fit_rule.py <dir with <machine>/results.csv> [<dir of another run>...]"""
import csv
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
POW2 = [2 ** k for k in range(1, 16)]
PARS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]

# several runs: the minimum time of each implementation per point
points = {}
for root in sys.argv[1:]:
    for m in sorted(Path(root).iterdir()):
        if not (m / "results.csv").exists():
            continue
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
    d["where"].append(f"{mesh[7:-4]} {layout} 1/{fraction}")
    d["size"].append(SIZES[mesh][KIND(fn)])
    d["count"].append(count)
    d["blocks"].append(count / fill if fill else 0)
    d["t"].append(t)


def choose(d, dense, scattered, par):
    serial = (d["count"] * dense < d["size"]) & (d["blocks"] * scattered < d["size"])
    parallel = ~serial & (d["count"] * par <= d["size"]) if par else np.zeros_like(serial)
    return np.where(serial, 1, np.where(parallel, 2, 0))


def ratios(d, choice):
    chosen = d["t"][np.arange(len(choice)), choice]
    best = d["t"].min(axis=1)
    return chosen / best, chosen / (best + ABS_MS)


print(f"{'function':44} {'current d/s/p':>16} {'worst':>6} | {'proposed d/s/p':>16} {'worst':>6} {'mean':>6}  worst row")
proposed = {}
for fn, d in data.items():
    d = {k: np.array(v) for k, v in d.items()}
    hasPar = np.isfinite(d["t"][:, 2]).all()
    best = None
    for dense in POW2:
        for scattered in POW2:
            if scattered < dense:
                continue
            for par in (p for p in PARS if p <= dense) if hasPar else [None]:
                _, adj = ratios(d, choose(d, dense, scattered, par))
                key = (round(adj.max(), 2), np.log(adj).mean())
                if best is None or key < best[0]:
                    best = (key, (dense, scattered, par))
    proposed[fn] = best[1]
    cur = CURRENT.get(fn)
    curWorst = ratios(d, choose(d, *cur))[1].max() if cur and (hasPar or not cur[2]) else np.nan
    raw, adj = ratios(d, choose(d, *best[1]))
    i = adj.argmax()
    fmt = lambda x: "/".join(str(v) for v in x if v) if x else "-"
    print(f"{fn:44} {fmt(cur):>16} {curWorst:6.2f} | {fmt(best[1]):>16} {adj.max():6.2f} {np.exp(np.log(adj).mean()):6.3f}  {d['machine'][i]} {d['where'][i]}")

print("\nWorst ratio chosen/fastest per machine with the proposed limits (raw, no noise slack):")
machines = sorted(set(m for d in data.values() for m in d["machine"]))
print(f"{'function':44} " + " ".join(f"{m:>15}" for m in machines))
for fn, d in data.items():
    d = {k: np.array(v) for k, v in d.items()}
    raw, adj = ratios(d, choose(d, *proposed[fn]))
    cells = []
    for m in machines:
        sel = d["machine"] == m
        cells.append(f"{raw[sel].max():6.2f} ({adj[sel].max():.2f})")
    print(f"{fn:44} " + " ".join(f"{c:>15}" for c in cells))
