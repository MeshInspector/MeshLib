#!/usr/bin/env python3
# DIAGNOSTIC ONLY (not for merge). Traces the OpenVDB -> Boost.IOStreams ->
# _FILE_OFFSET_BITS chain and the lean-PCH macro parity in this CI image, to
# explain why a lean GCC PCH reused across targets does or does not hit
# -Werror=invalid-pch. Prints to stdout; never fails the step.
import json, os, subprocess

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=180).stdout
    except Exception as e:
        return f"<error: {e}>"

print("########## 1. baked libopenvdb.so -> does it link boost_iostreams? ##########", flush=True)
sos = [s for s in sh("find / -xdev -name 'libopenvdb.so*' 2>/dev/null").split() if s]
if not sos:
    print("  (no libopenvdb.so found)")
for so in sorted(set(os.path.realpath(s) for s in sos)):
    needed = sh(f"objdump -p '{so}' 2>/dev/null | grep NEEDED")
    print(f"--- {so}")
    print(f"    boost_iostreams in NEEDED: {'YES (delayed-loading)' if 'boost_iostreams' in needed else 'no'}")
    for line in needed.splitlines():
        if 'boost' in line.lower() or 'iostreams' in line.lower():
            print("   ", line.strip())

print("\n########## 2. libboost_iostreams present in image? ##########", flush=True)
print(sh("find / -xdev -name 'libboost_iostreams*' 2>/dev/null") or "  (none)")

print("\n########## 3. OPENVDB_USE_DELAYED_LOADING define in openvdb headers? ##########", flush=True)
print(sh("grep -rln 'OPENVDB_USE_DELAYED_LOADING' "
         "$(find / -xdev -path '*openvdb*' -name '*.h' 2>/dev/null) 2>/dev/null | head")
      or "  (not found in headers)")

print("\n########## 4. compile_commands.json macro parity per target ##########", flush=True)
ccs = [c for c in sh("find . -name compile_commands.json 2>/dev/null").split() if c]
print("  compile_commands.json:", ccs or "(none yet -- pre-build run)")
want = ('MRPch', 'MRVoxels', 'MRMesh', 'MRViewer', 'MRPlugins', 'MRDental', 'MRInspector')
if ccs:
    try:
        d = json.load(open(ccs[0]))
        seen = {}
        for e in d:
            f = e.get('file', '')
            cmd = e.get('command') or ' '.join(e.get('arguments', []))
            for w in want:
                if f"/{w}/" in f and w not in seen:
                    seen[w] = dict(
                        DELAYED='OPENVDB_USE_DELAYED_LOADING' in cmd,
                        FILE_OFFSET_BITS='_FILE_OFFSET_BITS' in cmd,
                        BOOST_IOSTREAMS='BOOST_IOSTREAMS' in cmd,
                        reuses_pch=('cmake_pch' in cmd),
                    )
        for w in want:
            print(f"  {w:12}: {seen.get(w, '(no TU in this build)')}")
    except Exception as ex:
        print("  parse error:", ex)
print("\n########## diag done ##########", flush=True)
