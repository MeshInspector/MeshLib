# macOS Intel (x86_64) cross build on Apple Silicon

Builds the **Intel (`x86_64`)** macOS target of MeshLib on an **Apple Silicon (`arm64`)** self-hosted
runner, using a **native arm64 toolchain that cross-targets x86_64**: cmake/ninja/clang run natively
(fast compiles) and emit x86_64 via `-arch x86_64`, linking the x86_64 Homebrew at `/usr/local`. The
binaries run on Intel Macs — and on the build host under Rosetta, which is how CI tests them.

## Where it lives in CI

- The `x64-cross` leg of the `macos-build-test` job in
  [`build-test-macos.yml`](../../.github/workflows/build-test-macos.yml). The matrix is
  [`matrix/macos-config.json`](../../.github/workflows/matrix/macos-config.json);
  [`config.yml`](../../.github/workflows/config.yml) (step `set-macos-matrix`) selects it and **drops the
  `x64-cross` leg when the PR carries the `disable-build-macos-crossplatform` label**, so a down or busy
  self-hosted runner can never hang the hosted macOS legs (`disable-build-macos` skips all of macOS).
  The cross-only steps — the shim setup, `-DMR_PLATFORM=APPLE_x86_64`, `Verify x86_64 output` — are
  gated on `matrix.cross-osx-arch`, which only that leg sets.
- Runs on a runner labelled `[self-hosted, macos, arm64, crossplatform-build]`, provisioned by
  [`provision-runner.sh`](provision-runner.sh).
- Produces `meshlib_x64-cross.pkg`; [`test-distribution.yml`](../../.github/workflows/test-distribution.yml)
  installs and smoke-tests it on a real Intel Mac. That leg is gated by its `test_macos_crossplatform`
  input (= `build_enable_macos_crossplatform` from `config.yml`), so disabling the cross build also
  disables the distro test that would otherwise look for a `.pkg` that was never published.

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

## What still runs under Rosetta

Compilation is native; Rosetta is used for four things, only the last of which is fundamental:

1. **The `/usr/local` Homebrew itself.** Its Portable Ruby is an x86_64 binary, so every `brew` call
   there (install, `--prefix`, config) is translated, as are formula post-install hooks.
2. **Configure-time execution of x86_64 programs:** `python3.10-config` / CMake's FindPython (the
   `/usr/local` interpreter is x86_64) and CMake `try_run` probes.
3. **mrbind and the bindings.** They use the `llvm-pgo` keg, which on this leg is the **x86_64** keg
   in `/usr/local/Cellar` (see below), so `mrbind` and the bindings compiler are translated.
   `scripts/mrbind/generate.mk` has no macOS target-arch flag either; it yields an x86_64
   `mrmeshpy.so` because the x86_64 GNU `make` from `/usr/local` is first on `PATH` and its children
   inherit the translated execution. It works, but it is not native — a candidate follow-up.
4. **Running the Intel output** for `MRTest`, `MRTestC2`, the MeshViewer smoke test and the Python
   tests. Intel code cannot run on Apple Silicon hardware any other way.

A fully Rosetta-free runner would therefore be build-only (different x86_64 dependency source, a
toolchain file with pre-seeded `try_run` results) with all testing on real Intel hardware.

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
5. **A translated parent makes every child translated.** Anything started from an x86_64 process
   (the `/usr/local` `make`, a translated shell) runs `/usr/bin/clang++` as x86_64 too. Keep the
   compile driven by the native `ninja` (gotcha 1); this is also why the bindings step is translated.
6. **The LLVM keg follows the Homebrew prefix, not the host CPU.** The shim makes `brew --prefix`
   report `/usr/local`, so `LLVM_PREFIX` resolves to the Intel keg there, which is the right one: the
   bindings are x86_64 and Homebrew ships no Intel `lld` any more. `install-llvm-pgo-keg` therefore
   selects its published asset by prefix (`/usr/local` → x86_64, `/opt/homebrew` → arm64) rather than
   by `uname -m`, which on this runner reports the arm64 *host*.

## Provisioning a runner

Two accounts are involved: an **administrator** (has sudo) and the CI **service account** (`runner`, no
sudo, runs the jobs). Run [`provision-runner.sh`](provision-runner.sh) as the service account; wherever
root is required it prints the exact one-time command for the administrator and exits 1:

1. **Rosetta 2** — `sudo softwareupdate --install-rosetta --agree-to-license`. Needed *before* the first
   `/usr/local` brew command (brew's Ruby there is x86_64).
2. **`/usr/local` skeleton owned by the service account** — `sudo mkdir -p /usr/local/{…}` plus
   `sudo chown -R runner:staff /usr/local/{…}` over Homebrew's directory set (the script prints the
   full list). `/usr/local` itself stays `root:wheel`.

The service account's re-run then `git clone`s Homebrew into `/usr/local/Homebrew`, links
`/usr/local/bin/brew` and runs `brew update --force --quiet`. The official installer is deliberately
not used: since 2026-09 it is Apple-Silicon-only (it aborts on an x86_64 `uname`, and natively it only
targets `/opt/homebrew`) and it hard-requires sudo. No `arch -x86_64` is needed anywhere: brew at
`/usr/local` selects its x86_64 Ruby by prefix and serves Intel bottles regardless of the caller's
architecture (`/usr/local/bin/brew config` reports `macOS: …-x86_64`), which is exactly how the CI shim
(`exec /usr/local/bin/brew`) invokes it. `--prewarm` installs the `requirements/macos.txt` formulae and
the binding-generation deps up front.

## Support horizon

Homebrew 7.0 (2026-09-13) moved Intel macOS to **Tier 3: no new Intel bottles**. Existing bottles keep
installing, but an updated formula may build from source — under Rosetta on this runner, which for
`llvm@22` means hours — and Homebrew intends to stop running on Intel in or after September 2027. This
applies equally to the GitHub-hosted `macos-15-intel` leg, which installs the same Intel bottles.
Consequences:

- Keep the runner's `/usr/local` formulae pinned: CI already sets `HOMEBREW_NO_AUTO_UPDATE=1` and
  `HOMEBREW_NO_INSTALL_UPGRADE=1`; never `brew upgrade` that prefix by hand.
- `brew doctor` on `/usr/local` prints an expected Tier-3 notice; it is not an error.
- The Intel target as a whole has a bounded life; plan its retirement alongside the runner.

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

Everything else is CI wiring (the matrix JSON, the `config.yml` gate, the runner shim in the
workflow); the remaining changes are:

| Change | File |
|---|---|
| Honor `-D HOMEBREW_PREFIX=<prefix>` (falls back to `brew --prefix`) and validate it | [`ConfigureHomebrew.cmake`](../../cmake/Modules/ConfigureHomebrew.cmake) |
| Forward the cross knobs (`CMAKE_OSX_ARCHITECTURES`, `CMAKE_MAKE_PROGRAM`, `HOMEBREW_PREFIX`) and honor a caller `NPROC` | [`build_source.sh`](../../scripts/build_source.sh), [`build_thirdparty.sh`](../../scripts/build_thirdparty.sh) |
| Key the thirdparty cache on the target arch | [`install-macos-thirdparty`](../../.github/actions/install-macos-thirdparty/action.yml) |
| Pick the published LLVM keg by Homebrew prefix instead of `uname -m` (gotcha 6) | [`install-llvm-pgo-keg`](../../.github/actions/install-llvm-pgo-keg/action.yml) |
