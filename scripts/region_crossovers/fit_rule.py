#!/usr/bin/env python3
"""Fit and check the walk/scan rule of MeshInspector/MeshLib#7134 on region_crossovers results.

Rule for a set of `count` elements in a mesh of `size` elements (faces, vertices or undirected edges):
    walk  iff  count * kc < size                          (limit for a dense, contiguous set)
          and  nonEmptyBlocks * ks < size                 (limit by the 64-bit blocks the set touches)
where nonEmptyBlocks = count / setFill. For a scattered set nonEmptyBlocks ~ count, so the second condition is the
scattered limit; for a clustered set it is relaxed by the fill.
kc and ks are fitted per function as the worst (largest k) crossover over all machines of contiguous and scattered
layouts; the ball layouts (spatially clustered, not contiguous) are only used to check the rule.
Usage: fit_rule.py <dir with <machine>/results.csv>"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

# element counts of the inputs: faces for face-set functions, vertices or undirected edges otherwise
SIZES = {"dragon-1M.stl": {"faces": 1025766, "verts": 512880, "edges": 1538654},
         "dragon-250k.stl": {"faces": 244026, "verts": 122005, "edges": 366039}}
KIND = lambda fn: "verts" if fn.endswith("(verts)") else "edges" if fn.endswith("(edges)") else "faces"

rows = []
for m in sorted(Path(sys.argv[1]).iterdir()):
    if (m / "results.csv").exists():
        for r in csv.DictReader(open(m / "results.csv")):
            if "setFill" not in r:
                continue
            r["machine"] = m.name
            rows.append(r)


def crossover(pts):
    best = None
    for k in sorted(pts, reverse=True):
        if pts[k][1] >= pts[k][0]:
            break
        best = k
    return best or 2048


curves = defaultdict(dict)
for r in rows:
    curves[(r["machine"], r["mesh"], r["layout"], r["function"])][int(r["fraction"])] = (float(r["scanMs"]), float(r["walkMs"]))
fns = list(dict.fromkeys(r["function"] for r in rows))
fit = {}
for fn in fns:
    kc = max(crossover(p) for (m, mesh, lay, f), p in curves.items() if f == fn and lay.endswith("contiguous"))
    ks = max(crossover(p) for (m, mesh, lay, f), p in curves.items() if f == fn and lay.endswith("scattered"))
    fit[fn] = (kc, ks)

print(f"{'function':44} {'kc':>5} {'ks':>5} | {'walks':>6} {'slower':>6} {'worst walk/scan':>16} | {'scans':>6} {'missed':>6} {'best missed':>12}")
for fn in fns:
    kc, ks = fit[fn]
    walks = slower = scans = missed = 0
    worst = (1.0, "")
    best_missed = 1.0
    for r in rows:
        if r["function"] != fn:
            continue
        size = SIZES[r["mesh"]][KIND(fn)]
        count, fill = int(r["setSize"]), float(r["setFill"])
        scan, walk = float(r["scanMs"]), float(r["walkMs"])
        blocks = count / fill if fill else 0
        if count * kc < size and blocks * ks < size:
            walks += 1
            if walk > scan * 1.1:  # 10% = measurement noise
                slower += 1
            if walk / scan > worst[0]:
                worst = (walk / scan, f"{r['machine']} {r['layout']} 1/{r['fraction']}")
        else:
            scans += 1
            if walk < scan / 1.1:
                missed += 1
                best_missed = min(best_missed, walk / scan)
    print(f"{fn:44} {'1/'+str(kc):>5} {'1/'+str(ks):>5} | {walks:6} {slower:6} {worst[0]:6.2f} {worst[1][:30]:30} | {scans:6} {missed:6} {best_missed:12.2f}")
