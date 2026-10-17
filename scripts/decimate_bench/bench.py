#!/usr/bin/env python3
"""Decimation comparison: MeshLib (run as in its MeshDecimate example) vs trueform (polydera).

Builds two small harnesses (ml/ and tf/), times both libraries alternately on trueform's own dragon meshes
(50k-1M triangles), checks the quality of the 1M outputs, and writes results/<timestamp>/summary.md.
Works on Windows, Linux and macOS; see README.md.
"""
import argparse
import csv
import datetime
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
IS_WIN = platform.system() == "Windows"
EXE = ".exe" if IS_WIN else ""
SIZES = ["50k", "125k", "250k", "500k", "750k", "1M"]
RATIOS = ["0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"]


def run(cmd, env=None, cwd=None, capture=False):
    cmd = [str(c) for c in cmd]
    if env is not None:  # Windows looks the program up in our own PATH, not the child's
        cmd[0] = shutil.which(cmd[0], path=env.get("PATH")) or cmd[0]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, cwd=cwd, text=True, stdout=subprocess.PIPE if capture else None)
    if r.returncode:
        sys.exit(f"command failed with exit code {r.returncode}: {' '.join(cmd)}")
    return r.stdout if capture else None


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=True).stdout.strip()


def windows_build_env():
    """Environment of a Visual Studio developer prompt for this machine (cl, cmake and ninja on PATH)."""
    arm = platform.machine().upper() in ("ARM64", "AARCH64")
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    vs = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                         "Microsoft.VisualStudio.Component.VC.Tools." + ("ARM64" if arm else "x86.x64"), "-property", "installationPath"],
                        text=True, capture_output=True, check=True).stdout.strip()
    vcvars = Path(vs) / ("VC/Auxiliary/Build/vcvarsarm64.bat" if arm else "VC/Auxiliary/Build/vcvars64.bat")
    out = subprocess.run(f'set "PATH={vswhere.parent};%PATH%" && "{vcvars}" >nul && set', shell=True,
                         text=True, capture_output=True, check=True).stdout
    # Windows variable names are case-insensitive; `set` prints "Path", lookups below use "PATH"
    env = {k.upper(): v for k, v in (line.split("=", 1) for line in out.splitlines() if "=" in line)}
    return env, Path(vs)


def find_clang_cl():
    for c in [shutil.which("clang-cl"), r"C:\Program Files\LLVM\bin\clang-cl.exe"]:
        if c and Path(c).exists():
            return c
    return None


def cpu_name():
    try:
        if IS_WIN:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, capture_output=True).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def cmake_build(src, build, defs, env, cxx=None):
    gen = ["-G", "Ninja"] if shutil.which("ninja", path=env.get("PATH")) else []
    args = ["cmake", "-S", src, "-B", build, *gen, "-DCMAKE_BUILD_TYPE=Release"]
    if cxx:
        # mimalloc, fetched by trueform's CMake, is C: pair a C compiler with the chosen C++ one
        name = Path(cxx).name
        cc = str(Path(cxx).with_name(name.replace("clang++", "clang").replace("g++", "gcc")))
        args += [f"-DCMAKE_CXX_COMPILER={cxx}", f"-DCMAKE_C_COMPILER={cc}"]
    args += [f"-D{k}={v}" for k, v in defs.items()]
    run(args, env=env)
    run(["cmake", "--build", build, "--config", "Release", "--parallel"], env=env)
    exe = next(p for p in [build / f"{build.name}_bench{EXE}", build / "Release" / f"{build.name}_bench{EXE}"] if p.exists())
    return exe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ml = ap.add_mutually_exclusive_group(required=True)
    ml.add_argument("--meshlib-source", type=Path, help="MeshLib checkout built with MSBuild (Windows)")
    ml.add_argument("--meshlib-prefix", type=Path, help="installed MeshLib, or an unpacked release package (any OS)")
    ap.add_argument("--build-mrmesh", action="store_true", help="with --meshlib-source: build MRMesh Release first")
    ap.add_argument("--vcpkg", type=Path, default=os.environ.get("MESHLIB_VCPKG", r"D:\lib\vcpkg\installed\x64-windows-meshlib"),
                    help="Windows: vcpkg triplet directory with TBB, spdlog, fmt (default: %(default)s)")
    ap.add_argument("--trueform", type=Path, default=KIT / "trueform", help="trueform checkout, cloned on first run")
    ap.add_argument("--trueform-ref", help="check out this trueform commit or tag")
    ap.add_argument("--update-trueform", action="store_true", help="move trueform to its latest master")
    ap.add_argument("--tf-cxx", help="C++ compiler for trueform (default: clang-cl on Windows, clang++ elsewhere if found)")
    ap.add_argument("--parts", type=int, default=16, help="MeshLib subdivideParts (default: %(default)s, as in the example)")
    ap.add_argument("--rounds", type=int, default=2, help="interleaved rounds, minimum reported (default: %(default)s)")
    ap.add_argument("--reps", type=int, default=10, help="runs inside each measurement, minimum taken (default: %(default)s)")
    ap.add_argument("--skip-quality", action="store_true", help="timing only")
    ap.add_argument("--ml-define", action="append", default=[], metavar="KEY=VALUE",
                    help="extra CMake definition for the MeshLib harness, e.g. MRMESH_CONFIG_DIR=... (repeatable)")
    a = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out = KIT / "results" / stamp
    out.mkdir(parents=True, exist_ok=True)

    # --- sources ---
    tf_dir = a.trueform.resolve()
    if not (tf_dir / ".git").exists():
        run(["git", "clone", "-q", "https://github.com/polydera/trueform", tf_dir])
    if a.update_trueform or a.trueform_ref:
        run(["git", "-C", tf_dir, "fetch", "-q", "origin"])
        run(["git", "-C", tf_dir, "checkout", "-q", a.trueform_ref or "origin/HEAD"])
    tf_commit = git(tf_dir, "log", "-1", "--format=%h %cs")
    data = tf_dir / "benchmarks" / "data"

    # --- build ---
    env = dict(os.environ)
    vs = None
    if IS_WIN:
        env, vs = windows_build_env()
    ml_defs, ml_runtime = {}, []
    if a.meshlib_source:
        src = a.meshlib_source.resolve()
        ml_commit = git(src, "log", "-1", "--format=%h %cs")
        if not IS_WIN:
            sys.exit("--meshlib-source is for MSBuild checkouts on Windows; elsewhere install MeshLib "
                     "(cmake --install, a .deb/.pkg, or an unpacked release) and pass --meshlib-prefix")
        if a.build_mrmesh:
            run([vs / "MSBuild/Current/Bin/MSBuild.exe", src / "source/MRMesh/MRMesh.vcxproj", "/p:Configuration=Release",
                 "/p:Platform=x64", f"/p:SolutionDir={src / 'source'}{os.sep}", "/m", "/v:m", "/nologo"], env=env)
        v = a.vcpkg.resolve()
        incs = [src / "source", src / "thirdparty/parallel-hashmap", src / "thirdparty/expected/include", v / "include", v / "include/eigen3"]
        ml_defs = {"MESHLIB_INCLUDE_DIRS": ";".join(p.as_posix() for p in incs),
                   "MESHLIB_LIB_DIRS": ";".join(p.as_posix() for p in [src / "source/x64/Release", v / "lib"])}
        ml_runtime = [src / "source/x64/Release", v / "bin"]
    else:
        pre = a.meshlib_prefix.resolve()
        ml_commit = f"installed at {pre}"
        ml_defs = {"CMAKE_PREFIX_PATH": pre.as_posix()}
        if platform.system() == "Darwin":
            ml_defs["CMAKE_FRAMEWORK_PATH"] = pre.as_posix()
        ml_runtime = [pre / "lib", pre / "bin"]

    ml_defs.update(d.split("=", 1) for d in a.ml_define)

    build = KIT / "build"
    ml_exe = cmake_build(KIT / "ml", build / "ml", ml_defs, env)

    tf_defs = {"TRUEFORM_DIR": tf_dir.as_posix()}
    if IS_WIN:
        tf_defs["CMAKE_PREFIX_PATH"] = a.vcpkg.resolve().as_posix()  # TBB
    tf_cxx = a.tf_cxx or (find_clang_cl() if IS_WIN else shutil.which("clang++"))
    tf_exe = cmake_build(KIT / "tf", build / "tf", tf_defs, env, tf_cxx)

    # --- runtime library paths ---
    renv = dict(os.environ)
    var = "PATH" if IS_WIN else ("DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH")
    extra = ml_runtime + ([a.vcpkg.resolve() / "bin"] if IS_WIN else [])
    renv[var] = os.pathsep.join([str(p) for p in extra if p.exists()] + [renv.get(var, "")])

    def timed(exe, *args):
        line = run([exe, "time", *args], env=renv, capture=True).strip().splitlines()[-1]
        return [float(x) for x in line.split(",")]

    # --- timing: the two libraries alternate, so a slow patch of the machine hits both ---
    best = {}

    def keep(key, vals):
        best[key] = vals if key not in best else [min(x, y) for x, y in zip(best[key], vals)]

    cases = [(s, "0.5") for s in SIZES] + [("1M", q) for q in RATIOS if q != "0.5"]
    for r in range(1, a.rounds + 1):
        for s, q in cases:
            print(f"round {r}: {s} at {q}", flush=True)
            mesh = data / f"dragon-{s}.stl"
            keep(("tf", s, q), timed(tf_exe, mesh, q, a.reps))
            keep(("ml", s, q), timed(ml_exe, mesh, a.parts, q, a.reps))
    with open(out / "timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lib", "mesh", "ratio", "prep_ms", "decimate_ms", "faces"])
        for (lib, s, q), v in sorted(best.items()):
            w.writerow([lib, s, q, *v])

    # --- quality of the 1M outputs at 50% and 10%, measured by MeshLib against the original ---
    quality = []
    if not a.skip_quality:
        orig = data / "dragon-1M.stl"
        for q in ["0.5", "0.1"]:
            print(f"quality at {q}", flush=True)
            run([tf_exe, "save", orig, q, "-1", out / f"tf-{q}.stl"], env=renv)
            run([ml_exe, "save", orig, a.parts, q, out / f"ml-{q}.mrmesh"], env=renv)
            for lib, res in [("trueform", out / f"tf-{q}.stl"), ("MeshLib", out / f"ml-{q}.mrmesh")]:
                rows = list(csv.DictReader(run([ml_exe, "eval", orig, res], env=renv, capture=True).strip().splitlines()))
                quality.append({"lib": lib, "ratio": q, **rows[0]})
        with open(out / "quality.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(quality[0]))
            w.writeheader()
            w.writerows(quality)

    # --- summary ---
    def t(lib, s, q):
        return best[(lib, s, q)]

    md = [f"# Decimation: MeshLib vs trueform ({stamp})", "",
          f"- Machine: {cpu_name()}, {platform.system()} {platform.version() if IS_WIN else platform.release()}",
          f"- MeshLib: {ml_commit}; packOptimally(false), then decimateMesh with subdivideParts = {a.parts} (as in the MeshDecimate example)",
          f"- trueform: {tf_commit}; default decimate_config (quality check off), built with {Path(tf_cxx).name if tf_cxx else 'the default compiler'} -O3 -march=native + mimalloc",
          f"- Times: minimum over {a.rounds} interleaved rounds x {a.reps} runs, in ms", "",
          "## Decimation to 50% by mesh size", "",
          "| Triangles | trueform decimate | MeshLib decimate | trueform incl. half-edge build | MeshLib incl. packOptimally | MeshLib speed-up (decimate / incl. prep) |",
          "| --- | --- | --- | --- | --- | --- |"]
    for s in reversed(SIZES):
        tv, mv = t("tf", s, "0.5"), t("ml", s, "0.5")
        md.append(f"| {s} | {tv[1]:.0f} | {mv[1]:.0f} | {tv[0] + tv[1]:.0f} | {mv[0] + mv[1]:.0f} | "
                  f"{tv[1] / mv[1]:.2f}x / {(tv[0] + tv[1]) / (mv[0] + mv[1]):.2f}x |")
    md += ["", "## 1M triangles by target ratio (decimation only)", "",
           "| Target | trueform | MeshLib | MeshLib speed-up |", "| --- | --- | --- | --- |"]
    for q in RATIOS:
        tv, mv = t("tf", "1M", q), t("ml", "1M", q)
        md.append(f"| {q} | {tv[1]:.0f} | {mv[1]:.0f} | {tv[1] / mv[1]:.2f}x |")
    if quality:
        md += ["", "## Output quality on 1M (measured with MeshLib against the original mesh)", "",
               "| Library | Target | Flipped vs original | Normal off >60 deg | Self-intersecting pairs | Zero-area | Aspect >100 | Aspect >1000 | Mean dist to orig (x diag) |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in quality:
            md.append(f"| {r['lib']} | {r['ratio']} | {r['flipped']} | {r['normal_off_gt60']} | {r['self_intersecting_pairs']} | "
                      f"{r['zero_area']} | {r['aspect_gt100']} | {r['aspect_gt1000']} | {r['dist_to_orig_mean']} |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nResults: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
