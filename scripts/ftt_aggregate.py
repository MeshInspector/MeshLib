#!/usr/bin/env python3
# Aggregate clang -ftime-trace JSON files in a directory and print per-category totals.
# Sums the per-TU "Total <Category>" roll-up events (durations in microseconds) across
# all translation units, so we can see where compile time goes (header parsing vs
# template instantiation vs codegen) summed over the whole bindings build.
import sys, json, glob, os, collections

d = sys.argv[1] if len(sys.argv) > 1 else "."
tot = collections.defaultdict(float)
n = 0
for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
    try:
        j = json.load(open(f, encoding="utf-8", errors="ignore"))
    except Exception:
        continue
    evs = j.get("traceEvents")
    if not evs:
        continue
    hit = False
    for e in evs:
        nm = e.get("name", "")
        if nm.startswith("Total "):
            tot[nm] += e.get("dur", 0)
            hit = True
    if hit:
        n += 1

print("FTT_FILES=%d  DIR=%s" % (n, d))
order = ["Total ExecuteCompiler", "Total Frontend", "Total Backend",
         "Total Source", "Total ParseClass",
         "Total InstantiateClass", "Total InstantiateFunction",
         "Total PerformPendingInstantiations",
         "Total CodeGen Function", "Total OptModule", "Total OptFunction",
         "Total CodeGenPasses", "Total RunPass"]
seen = set()
for k in order:
    if k in tot:
        print("FTT|%s|%.1f" % (k, tot[k] / 1e6)); seen.add(k)
for k in sorted(tot, key=lambda x: -tot[x]):
    if k not in seen and tot[k] > 3e6:
        print("FTT|%s|%.1f" % (k, tot[k] / 1e6))
