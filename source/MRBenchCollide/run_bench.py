#!/usr/bin/env python3
"""Interleaved A/B run of MRBenchCollide: master build in --bin-a, PR build in --bin-b.

Each round runs both builds once (order alternates). Per case the best time over all rounds is reported;
for the sphere Booleans, the best time per pose over rounds, then AVG/MEDIAN/MIN/MAX over poses.
"""
import argparse, os, platform, statistics, subprocess, sys

def cpu_model():
    try:
        if sys.platform.startswith('linux'):
            for line in open('/proc/cpuinfo'):
                if line.lower().startswith(('model name', 'hardware')):
                    return line.split(':', 1)[1].strip()
            return subprocess.run(['lscpu'], capture_output=True, text=True).stdout.split('Model name:')[1].splitlines()[0].strip()
        if sys.platform == 'darwin':
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
        if sys.platform == 'win32':
            out = subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name'], capture_output=True, text=True).stdout
            return out.strip()
    except Exception as e:
        return f'unknown ({e})'
    return platform.processor()

def run_once(bin_dir, exe, extra_lib_path, args):
    env = dict(os.environ)
    path = os.path.join(bin_dir, exe + ('.exe' if sys.platform == 'win32' else ''))
    libs = os.pathsep.join([bin_dir] + ([extra_lib_path] if extra_lib_path else []))
    if sys.platform.startswith('linux'):
        env['LD_LIBRARY_PATH'] = libs + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    elif sys.platform == 'win32':
        env['PATH'] = libs + os.pathsep + env.get('PATH', '')
    p = subprocess.run([path] + args, env=env, capture_output=True, text=True)
    sys.stderr.write(p.stderr)
    if p.returncode != 0 or 'E done' not in p.stdout:
        sys.stdout.write(p.stdout)
        raise SystemExit(f'{path} failed with code {p.returncode}')
    return p.stdout

def parse(out, res):
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'V':
            res.setdefault('variant', set()).add(parts[1])
        elif parts[0] == 'T':
            res.setdefault('T', {}).setdefault(parts[1], []).append(float(parts[2]))
        elif parts[0] == 'H':
            res.setdefault('H', {}).setdefault(parts[1], set()).add(parts[2])
        elif parts[0] == 'P':
            res.setdefault('P', {}).setdefault(parts[1], []).append([float(x) for x in parts[2:]])
        elif parts[0] == 'W':
            res.setdefault('W', []).append(line)

def pct(a, b):
    return f'{(b - a) / a * 100:+.1f}%' if a > 0 else 'n/a'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', action='append', required=True, help='name=dir; the first is the baseline')
    ap.add_argument('--exe', default='MRBenchCollide')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--extra-lib-path', default='')
    ap.add_argument('--title', default=platform.platform())
    ap.add_argument('--out', default='bench_summary.md')
    ap.add_argument('bench_args', nargs='*')
    a = ap.parse_args()

    bins = [b.split('=', 1) for b in a.bin]
    names = [n for n, _ in bins]
    res = {n: {} for n in names}
    for r in range(a.rounds):
        order = bins[r % len(bins):] + bins[:r % len(bins)]
        for n, d in order:
            print(f'round {r + 1}/{a.rounds}: {n}', flush=True)
            parse(run_once(d, a.exe, a.extra_lib_path, a.bench_args), res[n])

    base = names[0]
    lines = [f'## {a.title}', '', f'CPU: {cpu_model()}, {os.cpu_count()} logical CPUs; {a.rounds} interleaved rounds per build.', '']
    badVariant = [n for n in names if res[n].get('variant') != {n}]
    if badVariant:
        lines += [f'**WARNING: wrong variant loaded for {badVariant}: ' + str({n: res[n].get('variant') for n in names}) + '**', '']
    mism = sorted({k for n in names[1:] for k in res[base].get('H', {}) if res[base]['H'][k] != res[n].get('H', {}).get(k)})
    nondet = sorted({k for n in names for k, s in res[n].get('H', {}).items() if len(s) > 1})
    lines.append(f'Result hashes: {"all identical" if not mism else "DIFFER in " + ", ".join(mism)}'
                 + (f'; nondeterministic across rounds: {", ".join(nondet)}' if nondet else '') + '.')
    for n in names:
        for w in res[n].get('W', []):
            lines.append(f'- {n}: {w}')
    lines.append('')

    lines += ['| case | ' + ' | '.join(f'{n}, ms' for n in names) + ' |', '|---' * (len(names) + 1) + '|']
    def row(k, vals):
        cells = [f'{vals[0]:.2f}'] + [f'{v:.2f} ({pct(vals[0], v)})' for v in vals[1:]]
        lines.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    for k in res[base].get('T', {}):
        row(k, [min(res[n]['T'].get(k, [float('nan')])) for n in names])
    for k in res[base].get('P', {}):
        per = {n: [min(x) for x in zip(*res[n]['P'][k])] for n in names}
        for nm, f in (('AVG', statistics.fmean), ('MEDIAN', statistics.median), ('MIN', min), ('MAX', max)):
            row(f'{k} {nm}', [f(per[n]) for n in names])
    lines.append('')

    text = '\n'.join(lines)
    print(text)
    with open(a.out, 'w') as f:
        f.write(text + '\n')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write(text + '\n')
    if mism or badVariant:
        raise SystemExit('variant or result mismatch')

if __name__ == '__main__':
    main()
