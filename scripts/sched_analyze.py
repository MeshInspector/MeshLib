#!/usr/bin/env python3
# Read a sampler file of "<epoch> <num_running_compiler_procs>" lines and report
# core-occupancy stats for a (4-core) build: utilization, time at each parallelism
# level, idle (<=1) vs saturated (>=4) seconds.
import sys, collections
cores = 4
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sched.txt"
ts = []
for line in open(path):
    p = line.split()
    if len(p) == 2:
        try: ts.append((int(p[0]), int(p[1])))
        except ValueError: pass
n = len(ts)
if not n:
    print("SCHED no samples"); sys.exit(0)
busy = sum(min(b, cores) for _, b in ts)
ideal = cores * n
hist = collections.Counter(b for _, b in ts)
print(f"SCHED samples={n}s util={100*busy/ideal:.0f}% busy={busy} ideal={ideal} core-s")
for k in sorted(hist):
    print(f"SCHED par={k}: {hist[k]}s")
print(f"SCHED idle(par<=1)={sum(1 for _,b in ts if b<=1)}s  saturated(par>={cores})={sum(1 for _,b in ts if b>=cores)}s")
# wasted core-seconds vs a perfectly-parallel build of the same work
print(f"SCHED wasted_core_s={ideal-busy}  (if fully parallel, wall ~= {busy/cores:.0f}s vs actual {n}s)")
