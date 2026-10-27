# Interleaved A/B of bool_bench.py: python ab.py <rounds> <name>=<python exe> ...
# The last variant is compared to the first one.
import os
import platform
import statistics
import subprocess
import sys
from collections import defaultdict

bench = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bool_bench.py")
rounds = int(sys.argv[1])
variants = [a.split("=", 1) for a in sys.argv[2:]]
times = defaultdict(lambda: defaultdict(list))
print(f"{platform.platform()} {platform.machine()} cpus={os.cpu_count()} python={platform.python_version()}")

for r in range(rounds):
    for name, exe in variants:
        p = subprocess.run([exe, bench, "3"], capture_output=True, text=True)
        if p.returncode:
            sys.exit(f"{name} failed:\n{p.stdout}\n{p.stderr}")
        out = p.stdout
        for line in out.splitlines():
            if line.startswith("mimalloc"):
                if r == 0:
                    print(f"{name}: {line}")
                continue
            case, ms = line.split()
            times[case][name].append(float(ms))
    print(f"round {r + 1}/{rounds} done", flush=True)

names = [n for n, _ in variants]
a, b = names[0], names[-1]
lines = [f"| case | {a} min | {a} med | {b} min | {b} med | {b}/{a} min | {b}/{a} med |", "|---|---|---|---|---|---|---|"]
for case, per in times.items():
    lines.append(f"| {case} | {min(per[a]):.1f} | {statistics.median(per[a]):.1f} | {min(per[b]):.1f} | "
                 f"{statistics.median(per[b]):.1f} | {min(per[b]) / min(per[a]):.3f} | "
                 f"{statistics.median(per[b]) / statistics.median(per[a]):.3f} |")
print("\n".join(lines))
for case, per in times.items():
    print(case, {n: per[n] for n in names})
summary = os.environ.get("GITHUB_STEP_SUMMARY")
if summary:
    with open(summary, "a") as f:
        f.write(f"### {os.environ.get('RUNNER_LABEL', '')} ({platform.machine()}, {os.cpu_count()} cpus)\n\n" + "\n".join(lines) + "\n")
