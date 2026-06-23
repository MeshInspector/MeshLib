#!/usr/bin/env python3
# Aggregate clang -ftime-trace InstantiateClass/InstantiateFunction events by template
# name across all JSON files in a dir, to see WHICH templates dominate instantiation
# (MeshLib types vs libstdc++ std::* vs pybind11 wrappers).
import sys, json, glob, os, collections
d = sys.argv[1] if len(sys.argv) > 1 else "."
byname = collections.defaultdict(float)
bycat = collections.defaultdict(float)
nfiles = 0; total = 0.0

def cat(detail):
    for key in ("pybind11", "std::", "tl::expected", "phmap", "boost", "Eigen", "fmt::", "MR::"):
        if key in detail:
            return key
    return "other"

for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
    try:
        j = json.load(open(f, encoding="utf-8", errors="ignore"))
    except Exception:
        continue
    ev = j.get("traceEvents")
    if not ev:
        continue
    nfiles += 1
    for e in ev:
        if e.get("ph") == "X" and e.get("name") in ("InstantiateClass", "InstantiateFunction"):
            dur = e.get("dur", 0)
            det = (e.get("args") or {}).get("detail", "?")
            byname[det] += dur; total += dur; bycat[cat(det)] += dur

if total == 0:
    print("FTN no instantiation events found"); sys.exit(0)
print(f"FTN files={nfiles} total_instantiation_dur={total/1e6:.0f}s (nested events double-count)")
print("FTN --- by category (share of instantiation time) ---")
for k, v in sorted(bycat.items(), key=lambda x: -x[1]):
    print(f"FTN cat {k}: {v/1e6:.0f}s ({100*v/total:.0f}%)")
print("FTN --- top 30 instantiations by total dur ---")
for det, v in sorted(byname.items(), key=lambda x: -x[1])[:30]:
    print(f"FTN {v/1e6:7.1f}s  {det[:140]}")
