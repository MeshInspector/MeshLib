#!/usr/bin/env python3
# Drops the per-export JS of the C bindings from an Emscripten module (`var _MR_a,_MR_b,...;` and
# `_MR_a=Module["_MR_a"]=wasmExports["MR_a"];` for every export: 6 MB of glue for 40k functions).
# The generated JS takes them from `wasmExports` by name instead, which `-lexports.js` keeps unminified.
# The runtime's own exports (`_malloc`, ...) stay declared. Usage: strip_export_glue.py <module.mjs>
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8", newline="") as f:
    src = f.read()
size = len(src)

assignment = re.compile(r'_MR_\w+=Module\["_MR_\w+"\]=wasmExports\["MR_\w+"\];')
src, assignments = assignment.subn("", src)
if assignments == 0:
    if "createExportWrapper(" in src or "makeInvalidEarlyAccess(" in src:
        print(f"{path}: a debug (ASSERTIONS) build, the export glue is left as it is")
        sys.exit(0)
    sys.exit(f"{path}: no `_MR_*` export assignments with unminified names, is the module linked with `-lexports.js`?")
src, declarations = re.subn(r"\b_MR_\w+,", "", src)
src, last = re.subn(r",_MR_\w+;", ";", src)
declarations += last
if declarations != assignments or "_MR_" in src:
    sys.exit(f"{path}: {assignments} export assignments but {declarations} declarations removed, "
             f"and {src.count('_MR_')} `_MR_` left: the glue has an unexpected form")

with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(src)
print(f"{path}: dropped the glue of {assignments} exports, {size:,} -> {len(src):,} bytes")
