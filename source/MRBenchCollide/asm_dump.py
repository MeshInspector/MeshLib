#!/usr/bin/env python3
"""Recompiles one file of compile_commands.json to assembly and prints counts of arm64 barrier/acquire instructions."""
import json, os, re, shlex, subprocess, sys

db, src, out = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])
entry = next(e for e in json.load(open(db)) if os.path.normcase(os.path.abspath(e['file'])).endswith(os.path.normcase(os.path.normpath(src))))
args = entry.get('arguments') or shlex.split(entry['command'], posix=(os.name != 'nt'))
msvc = os.path.basename(args[0]).lower().strip('"') in ('cl', 'cl.exe', 'clang-cl', 'clang-cl.exe')
res = []
skip = False
for a in args:
    if skip:
        skip = False
        continue
    if msvc and (a.startswith('/Fo') or a.startswith('-Fo')):
        continue
    if not msvc and a == '-o':
        skip = True
        continue
    res.append(a)
if msvc:
    res += ['/FAs', '/Fa' + out, '/Fo' + out + '.obj']
else:
    res += ['-S', '-o', out]
if os.name == 'nt' and 'command' in entry:
    # keep the original quoting: CreateProcess receives the command line as is
    cmd = re.sub(r'\s[/-]Fo\S+', ' ', entry['command'])
    cmd += f' /FAs "/Fa{out}" "/Fo{out}.obj"' if msvc else f' -S -o "{out}"'
    cmd = re.sub(r'\s-o\s+\S+\.obj', ' ', cmd) if not msvc else cmd
    res = cmd
r = subprocess.run(res, cwd=entry['directory'], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-3000:], r.stderr[-3000:])
    sys.exit(r.returncode)
text = open(out, errors='replace').read()
if os.path.exists(out + '.obj'):
    os.remove(out + '.obj')
counts = {k: len(re.findall(r'^\s*' + k + r'\b', text, re.M | re.I)) for k in ('dmb', 'ldar', 'ldarb', 'ldapr', 'ldaprb', 'casal', 'casb', 'stlrb')}
print(f'asm {os.path.basename(out)}: ' + ', '.join(f'{k}={v}' for k, v in counts.items()))
