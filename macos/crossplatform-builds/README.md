# macOS cross-platform builds (x86_64 on Apple Silicon)

Build the **Intel (`x86_64`)** macOS target of MeshLib on **Apple Silicon (`arm64`)**
runners, instead of relying on a native Intel machine (the GitHub-hosted
`macos-15-intel` runner).

This lets the self-hosted `arm64` fleet produce the Intel `.pkg` / wheels, removing
the dependency on increasingly scarce Intel macOS hardware.

## How it works (Approach: Rosetta 2 + x86_64 Homebrew)

We reproduce an Intel mac *on Apple Silicon* rather than truly cross-compiling:

1. A **parallel x86_64 Homebrew at `/usr/local`** (running under Rosetta 2) coexists
   with the native `arm64` Homebrew at `/opt/homebrew`.
2. Putting `/usr/local/bin` **first on `PATH`** makes `brew`, `cmake`, `ninja`,
   `python`, `pkg-config`, `make` and `grep` all resolve to their **x86_64** builds.
3. Because the `cmake` binary is itself x86_64 (run under Rosetta), it reports
   `CMAKE_SYSTEM_PROCESSOR=x86_64` — so the architecture-gated SIMD code paths in the
   vendored thirdparty libraries (c-blosc, OpenVDB, oneTBB, libjpeg-turbo, …) are
   selected correctly **without patching any submodule**. Its child compilers
   inherit the x86_64 slice, and the x86_64 Homebrew installs x86_64 bottles.

The result: **no `arch -x86_64` wrapping of build commands, no thirdparty-submodule
patches, and no mrbind ABI surgery.** Two explicit guardrails make intent clear and
robust:

- `CMAKE_OSX_ARCHITECTURES=x86_64` — pins the produced object code.
- `HOMEBREW_DIR=/usr/local` — points the mrbind binding generator at the x86_64 `llvm@22`.

> Why "be an Intel mac" instead of native-arm cross-compile: the alternative (keep the
> arm64 toolchain, pass `-arch x86_64`) leaves `CMAKE_SYSTEM_PROCESSOR=arm64`, which
> would force patching every `CMAKE_SYSTEM_PROCESSOR`-gated thirdparty `CMakeLists.txt`
> plus `brew --prefix` overrides and explicit mrbind target flags. The Rosetta approach
> avoids all of that.

## What was added

| Change | File | Notes |
|---|---|---|
| `x64-cross` matrix entry | [`.github/workflows/build-test-macos.yml`](../../.github/workflows/build-test-macos.yml) | Runs on `[self-hosted, macos, arm64, build]` **alongside** `macos-15-intel` for validation. Distinct cache (`instance: self-hosted-arm-x64`); `x64-cross` artifacts are kept separate and are **not consumed downstream yet**. |
| `Configure x86_64 cross-build environment` step | same | Conditional on `matrix.cross-osx-arch == 'x86_64'`. Prepends `/usr/local/{bin,sbin}` to `PATH` and sets `CMAKE_OSX_ARCHITECTURES` + `HOMEBREW_DIR`. |
| Honor `CMAKE_OSX_ARCHITECTURES` | [`scripts/build_source.sh`](../../scripts/build_source.sh), [`scripts/build_thirdparty.sh`](../../scripts/build_thirdparty.sh) | Injected into the CMake invocation when set. **No-op when unset** → native builds unaffected. |
| Honor a caller-set `NPROC` | same two scripts | Caps build parallelism (e.g. to limit heat on dev machines); defaults to all cores when unset. |

The cache keys already discriminate by `instance`, `compiler`, and the Homebrew prefix
hash, so the `/usr/local` (x86_64) caches never collide with the `/opt/homebrew` (arm64)
or the GitHub-hosted Intel caches.

## What's needed (runner prerequisite)

Each self-hosted `arm64` `build` runner must have, **once**:

1. **Rosetta 2** installed:
   ```bash
   softwareupdate --install-rosetta --agree-to-license
   ```
2. **x86_64 Homebrew bootstrapped at `/usr/local`** (this requires `sudo` once):
   ```bash
   arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   It coexists with the native `arm64` Homebrew at `/opt/homebrew`; nothing in
   `/opt/homebrew` is modified.

All the **formulae** (`requirements/macos.txt` + `pybind11`, and `make grep lld llvm@22`
for mrbind) are installed automatically by the workflow — only the Homebrew *bootstrap*
above is manual.

## Reproducing a build locally (Apple Silicon → x86_64)

After the one-time prerequisite above, from the repo root:

```bash
# 1. Enter the x86_64 toolchain environment (x86_64 brew first on PATH)
eval "$(/usr/local/bin/brew shellenv)"
export CMAKE_OSX_ARCHITECTURES=x86_64 HOMEBREW_DIR=/usr/local
export CMAKE_C_COMPILER=/usr/bin/clang CMAKE_CXX_COMPILER=/usr/bin/clang++
export NPROC=5                       # optional: cap parallelism / heat

# 2. Install x86_64 formulae (idempotent)
{ cat requirements/macos.txt; echo pybind11; } | xargs /usr/local/bin/brew install --quiet

# 3. thirdparty + MeshLib core (x86_64)
./scripts/build_thirdparty.sh
MESHLIB_BUILD_RELEASE=ON MESHLIB_BUILD_DEBUG=OFF \
  MR_CMAKE_OPTIONS="-D MESHLIB_BUILD_MRMESH_PY_LEGACY=OFF -D MR_CXX_STANDARD=23 -D MR_PCH_USE_EXTRA_HEADERS=ON" \
  ./scripts/build_source.sh

# 4. (optional) mrbind + python bindings (x86_64)
JOBS=5 ./scripts/mrbind/install_mrbind_macos.sh
PATH="/usr/local/opt/make/libexec/gnubin:/usr/local/opt/grep/libexec/gnubin:$PATH" CXX=/usr/bin/clang++ \
  make -f scripts/mrbind/generate.mk -B -j5 \
    PYTHON_PKGCONF_NAME=python-3.10-embed MESHLIB_SHLIB_DIR=build/Release/bin

# 5. Verify arch + run tests (x86_64 binaries auto-run under Rosetta)
file build/Release/bin/libMRMesh.dylib            # ... x86_64
./build/Release/bin/MRTest                         # 294/294 pass; loads x86_64 mrmeshpy.so
```

> The commands above intentionally omit an `arch -x86_64` prefix: the x86_64 `cmake`
> from `/usr/local` is enough (see "How it works"). Wrapping a step in
> `arch -x86_64 bash -lc '…'` is a harmless extra safety net if a tool ever defaults
> to the wrong slice.

## Validation status

Validated end-to-end on an Apple **M4** (macOS 15.7):

- thirdparty libs, MeshLib core (`libMRMesh`, `MeshViewer`, `MRTest`), the mrbind tool,
  and the python modules (`mrmeshpy`/`mrmeshnumpy`/`mrcudapy`) all build as **x86_64**;
  `libMRMesh.dylib` links the `/usr/local` (x86_64) Homebrew dylibs.
- **294/294** unit tests pass under Rosetta; `MeshViewer` start-and-exit passes;
  `import mrmeshpy` works under x86_64 `python3.10` (embedded and external).

To confirm on the **first CI run** (not exercised in local validation):

- `.pkg` dylib bundling (`scripts/macos_bundle_dylibs.py` / `delocate`) gathering the
  x86_64 dylibs.
- The C++/C example builds.
- C bindings (`MRTestC2`).

## Follow-ups

- Promote `x64-cross` to **replace** the `macos-15-intel` job once it's green
  (repoint the `x64` entry / drop the hosted Intel runner).
- Optionally mirror the same setup in [`pip-build.yml`](../../.github/workflows/pip-build.yml)
  for the macOS wheels — it shares the thirdparty/mrbind caches with this workflow.
