#!/usr/bin/env python3
"""Crossovers of region_crossovers.cpp results: for each machine, function and layout, the largest region
(as a fraction of the mesh) up to which the walk is faster than the scan at every measured size.
Usage: analyze.py <dir with <machine>/results.csv and <machine>/info.txt> > summary.md"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
machines = sorted(p for p in root.iterdir() if (p / "results.csv").exists())
rows = defaultdict(dict)  # (machine, mesh, layout, function) -> {fraction: (scan, walk)}
functions = []
for m in machines:
    with open(m / "results.csv", newline="") as f:
        for r in csv.DictReader(f):
            key = (m.name, r["mesh"], r["layout"], r["function"])
            rows[key][int(r["fraction"])] = (float(r["scanMs"]), float(r["walkMs"]))
            if r["function"] not in functions:
                functions.append(r["function"])


def crossover(points):
    """largest region 1/k (smallest k) such that the walk wins at k and at every larger k (smaller region)"""
    best = None
    for k in sorted(points, reverse=True):  # from the smallest region up
        scan, walk = points[k]
        if walk >= scan:
            break
        best = k
    return best


def fmt(k):
    return "none" if k is None else ("1/%d" % k)


print("# MRRegionBoundary: walk vs scan crossovers\n")
print("Largest region, as a fraction of the mesh, up to which the walk of the input set beats the parallel scan "
      "at every measured size (1/1024 ... 1/4). Smaller fraction = earlier crossover.\n")
print("## Machines\n")
for m in machines:
    info = (m / "info.txt").read_text(errors="replace").strip().replace("\n", "; ") if (m / "info.txt").exists() else ""
    print(f"- **{m.name}**: {info}")

meshes = sorted({k[1] for k in rows})
layouts = ["packed-contiguous", "packed-scattered", "unpacked-contiguous", "packed-ball", "packed-ball-half", "unpacked-ball"]
for mesh in meshes:
    print(f"\n## {mesh}\n")
    header = "| Function | " + " | ".join(f"{m.name}<br>{l.replace('-', ' ')}" for m in machines for l in layouts) + " | worst |"
    print(header)
    print("|" + " --- |" * (2 + len(machines) * len(layouts)))
    for fn in functions:
        cells, worst = [], 0
        for m in machines:
            for l in layouts:
                pts = rows.get((m.name, mesh, l, fn))
                k = crossover(pts) if pts else None
                cells.append(fmt(k) if pts else "-")
                if pts:
                    worst = max(worst, k or 10**9)
        print(f"| {fn} | " + " | ".join(cells) + f" | **{fmt(worst if worst < 10**9 else None)}** |")
