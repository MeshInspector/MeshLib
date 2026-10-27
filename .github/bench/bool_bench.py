# Times MeshLib booleans through the Python bindings; prints "<case> <best ms>" per case.
# Usage: python bool_bench.py [reps]
import ctypes
import os
import sys
import time

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
from meshlib import mrmeshpy as mm


def loaded_libs():
    if sys.platform == "win32":
        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.EnumProcessModules.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
        psapi.GetModuleFileNameExW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong]
        proc = k32.GetCurrentProcess()
        mods = (ctypes.c_void_p * 4096)()
        needed = ctypes.c_ulong()
        psapi.EnumProcessModules(proc, mods, ctypes.sizeof(mods), ctypes.byref(needed))
        buf = ctypes.create_unicode_buffer(1024)
        res = []
        for i in range(needed.value // ctypes.sizeof(ctypes.c_void_p)):
            psapi.GetModuleFileNameExW(proc, mods[i], buf, 1024)
            res.append(buf.value)
        return res
    if sys.platform == "darwin":
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        return [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
    with open("/proc/self/maps") as f:
        return list({line.split()[-1] for line in f if "/" in line})


mi_paths = [p for p in loaded_libs() if "mimalloc" in os.path.basename(p).lower()]
if mi_paths:
    mi = ctypes.CDLL(mi_paths[0])
    libc = ctypes.CDLL("ucrtbase" if sys.platform == "win32" else None)
    libc.malloc.restype = ctypes.c_void_p
    mi.mi_is_in_heap_region.argtypes = [ctypes.c_void_p]
    redirected = mi.mi_is_redirected() if hasattr(mi, "mi_is_redirected") else "n/a"
    print(f"mimalloc {os.path.basename(mi_paths[0])} loaded, redirected={redirected}, "
          f"malloc in mimalloc heap={mi.mi_is_in_heap_region(libc.malloc(64))}")
else:
    print("mimalloc not loaded")


def best_ms(f):
    best = 1e30
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best


def check(r):
    if not r.valid():
        sys.exit(f"boolean failed: {r.errorString}")


xf = mm.AffineXf3f(mm.Matrix3f.rotation(mm.Vector3f(1, 1, 1).normalized(), 0.3), mm.Vector3f(0.7, 0.1, 0.05))

sp = mm.SphereParams()
sp.radius = 1.0
sp.numMeshVertices = 3366
small1 = mm.makeSphere(sp)
small2 = mm.Mesh(small1)
small2.transform(xf)

big1 = mm.makeUVSphere(1.0, 1000, 1000)
big2 = mm.Mesh(big1)
big2.transform(xf)
sp.radius = 0.5
sp.numMeshVertices = 20000
cutter = mm.makeSphere(sp)
cutter.transform(mm.AffineXf3f.translation(mm.Vector3f(0.9, 0.05, 0.02)))


def small_union_x100():
    for _ in range(100):
        check(mm.boolean(small1, small2, mm.BooleanOperation.Union))


print(f"small_union_x100 {best_ms(small_union_x100):.1f}")
print(f"1M_minus_sphere {best_ms(lambda: check(mm.boolean(big1, cutter, mm.BooleanOperation.DifferenceAB))):.1f}")
print(f"1M_union {best_ms(lambda: check(mm.boolean(big1, big2, mm.BooleanOperation.Union))):.1f}")
