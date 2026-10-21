# throw-away: interleaved runs of MRBoolBench{Static,Shared,Glibc}, min and median per workload
import collections, os, statistics, subprocess, sys

bindir = sys.argv[1]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 7
variants = ["MRBoolBenchStatic", "MRBoolBenchShared", "MRBoolBenchGlibc"]

with open("/proc/cpuinfo") as f:
    model = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), "unknown (arm64?)")
print(f"cpu_model: {model}, nproc: {os.cpu_count()}", flush=True)
for v in variants:
    ldd = subprocess.run(["ldd", os.path.join(bindir, v)], capture_output=True, text=True).stdout
    print(v, "ldd mimalloc:", [l.strip() for l in ldd.splitlines() if "mimalloc" in l], flush=True)

res = collections.defaultdict(lambda: collections.defaultdict(list))
for r in range(rounds):
    for v in variants[r % 3:] + variants[:r % 3]:
        p = subprocess.run([os.path.join(bindir, v)], capture_output=True, text=True)
        if r == 0:
            print(v, p.stderr.strip().replace("\n", "; "), flush=True)
        p.check_returncode()
        for line in p.stdout.splitlines():
            k, t = line.split()
            res[k][v].append(float(t))

print(f"\n{'workload':24} {'variant':20} {'min ms':>10} {'median ms':>10} {'min vs Static':>14}")
for k, byv in res.items():
    base = min(byv["MRBoolBenchStatic"])
    for v in variants:
        ts = byv[v]
        print(f"{k:24} {v:20} {min(ts):10.1f} {statistics.median(ts):10.1f} {100 * (min(ts) / base - 1):+13.1f}%")
