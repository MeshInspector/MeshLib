# macOS Intel (x86_64) cross build on Apple Silicon

Builds the **Intel (`x86_64`)** macOS target of MeshLib on an **Apple Silicon (`arm64`)** self-hosted
runner, using a **native arm64 toolchain that cross-targets x86_64**: cmake/ninja/clang run natively
(fast compiles) and emit x86_64 via `-arch x86_64`, linking the x86_64 Homebrew at `/usr/local`. The
binaries run on Intel Macs — and on the build host under Rosetta, which is how CI runs their tests.

## Where it lives in CI

- Job `macos-build-test-crossplatform` in
  [`build-test-macos.yml`](../../.github/workflows/build-test-macos.yml) — a separate, gated copy of
  the macOS build/test steps for the single `x64-cross` config.
- Gated by `build_enable_macos_crossplatform` (in [`config.yml`](../../.github/workflows/config.yml)):
  on by default. Add the `disable-macos-crossplatform` PR label to skip just this job (e.g. when the
  self-hosted runner is down) so it can't hang the hosted macOS legs; `disable-macos` skips all macOS.
- Runs on a runner labelled `[self-hosted, macos, arm64, crossplatform-build]`, provisioned by
  [`provision-runner.sh`](provision-runner.sh).
- Produces `meshlib_x64-cross.pkg`; [`test-distribution.yml`](../../.github/workflows/test-distribution.yml)
  installs and smoke-tests it on a real Intel Mac.

> **Intent:** once proven, this replaces the GitHub-hosted `macos-15-intel` x64 leg (Intel runners are
> being retired). Until then both run, and both `.pkg`s are published (the cross one suffixed
> `-x64-cross`).

## How it works

- **cmake / ninja / clang run natively as arm64.** `CMAKE_OSX_ARCHITECTURES=x86_64` makes AppleClang
  emit x86_64 objects. `-D MR_PLATFORM=APPLE_x86_64` labels the binary correctly — otherwise it
  inherits the host's `CMAKE_SYSTEM_PROCESSOR` (`arm64`) and ships a wrong platform string.
- x86_64 dependencies come from the **x86_64 Homebrew at `/usr/local`** (coexisting with the native
  arm64 Homebrew). [`ConfigureHomebrew.cmake`](../../cmake/Modules/ConfigureHomebrew.cmake) honors
  `-D HOMEBREW_PREFIX=/usr/local`.

## Critical gotchas (why a naive attempt silently falls back to Rosetta)

1. **Force the arm64 ninja.** CMake's `find_program` searches `/usr/local/bin` by default and picks up
   the **x86_64** ninja, which spawns **x86_64 clang under Rosetta** — silently defeating the native
   build. Pass `-D CMAKE_MAKE_PROGRAM=<arm64-brew>/bin/ninja`. Verify with
   `vmmap <clang-pid> | grep "Code Type"` → must say `ARM64`, not `X86-64 (Translated)`.
2. **Point find_package at `/usr/local`** with `-D CMAKE_PREFIX_PATH=/usr/local` so Python, OpenSSL,
   etc. resolve their x86_64 copies.
3. **x86_64 Python vs native cmake PATH tension.** Resolve with a small PATH shim mapping
   `cmake`/`ninja` → the arm64 brew and `python3.10*` → `/usr/local` (x86_64). See the
   "Configure native x86_64 cross-build environment" step.
4. **`CMAKE_SYSTEM_PROCESSOR` stays `arm64`** (it reflects the host, since cmake is native). Harmless
   for MeshLib's own SIMD (gated on the target macros `__x86_64__`/`__aarch64__`); `MR_PLATFORM` is
   set explicitly to compensate for the label.
5. Configure-time `try_run` probes execute x86_64 test binaries, which the OS runs via Rosetta
   transparently. Only these brief probes touch Rosetta; the bulk compilation is native.

## Provisioning a runner

Run [`provision-runner.sh`](provision-runner.sh) once per host (see its header for prerequisites). It
ensures a native arm64 Homebrew (cmake + ninja), Rosetta 2, and an x86_64 Homebrew at `/usr/local`
with the `requirements/macos.txt` formulae (`--prewarm` also installs the binding-generation deps).

## Reproducing locally

```bash
SHIM=$(mktemp -d)
ln -sf "$(brew --prefix)/bin/cmake"     "$SHIM/cmake"    # native arm64 cmake/ninja
ln -sf "$(brew --prefix)/bin/ninja"     "$SHIM/ninja"
ln -sf /usr/local/bin/python3.10        "$SHIM/python3.10"       # x86_64 Python
ln -sf /usr/local/bin/python3.10-config "$SHIM/python3.10-config"

env -i HOME="$HOME" \
  PATH="$SHIM:$(brew --prefix)/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  MESHLIB_BUILD_RELEASE=ON MESHLIB_BUILD_DEBUG=OFF \
  CMAKE_C_COMPILER=/usr/bin/clang CMAKE_CXX_COMPILER=/usr/bin/clang++ \
  MR_CMAKE_OPTIONS="\
    -D CMAKE_MAKE_PROGRAM=$(brew --prefix)/bin/ninja \
    -D HOMEBREW_PREFIX=/usr/local \
    -D CMAKE_PREFIX_PATH=/usr/local \
    -D CMAKE_OSX_ARCHITECTURES=x86_64 \
    -D MR_PLATFORM=APPLE_x86_64 \
    -D MR_CXX_STANDARD=23 -D MR_PCH_USE_EXTRA_HEADERS=ON" \
  bash ./scripts/build_source.sh
```

Confirm the output arch with `lipo -archs build/Release/bin/libMRMesh.dylib` → `x86_64` (CI asserts
this). The thirdparty-from-source libraries build the same way (native tools + the same `-D` flags).

## Source changes this requires

Everything else is CI wiring (the workflow job, the `config.yml` gate, the runner shim in the
workflow); the only non-CI source changes are:

| Change | File |
|---|---|
| Honor `-D HOMEBREW_PREFIX=<prefix>` (falls back to `brew --prefix`) and validate it | [`ConfigureHomebrew.cmake`](../../cmake/Modules/ConfigureHomebrew.cmake) |
| Forward the cross knobs (`CMAKE_OSX_ARCHITECTURES`, `CMAKE_MAKE_PROGRAM`, `HOMEBREW_PREFIX`) and honor a caller `NPROC` | [`build_source.sh`](../../scripts/build_source.sh), [`build_thirdparty.sh`](../../scripts/build_thirdparty.sh) |
