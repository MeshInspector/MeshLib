# macOS Intel (x86_64) cross build on Apple Silicon

Builds the **Intel (`x86_64`)** macOS target of MeshLib on an **Apple Silicon (`arm64`)**
self-hosted runner with the **native arm64 toolchain cross-targeting x86_64**: cmake, ninja and
clang run natively and emit x86_64 (`CMAKE_OSX_ARCHITECTURES=x86_64`), the dependencies come from
vcpkg. The binaries run on Intel Macs, and on the build host under Rosetta, which is how CI
tests them. The Build step takes about 2 minutes there against 12-17 on the GitHub-hosted Intel
runner.

## Where it lives in CI

- The `x64-cross` leg of [`matrix/macos-config.json`](../../.github/workflows/matrix/macos-config.json),
  run by the `macos-build-test` job of
  [`build-test-macos.yml`](../../.github/workflows/build-test-macos.yml) on a runner labelled
  `[self-hosted, macos, arm64, crossplatform-build]`. The cross-only steps (the shim setup,
  `-DMR_PLATFORM=APPLE_x86_64`, `Verify x86_64 output`) are gated on `matrix.cross-osx-arch`.
- The PR label `disable-build-macos-crossplatform` drops the leg
  ([`config.yml`](../../.github/workflows/config.yml), step `set-macos-matrix`), so a down or busy
  runner cannot hang the hosted macOS legs. It also drops the distribution tests of the package.
- The leg produces `meshlib_x64-cross.pkg`;
  [`test-distribution.yml`](../../.github/workflows/test-distribution.yml) installs it on Intel
  macOS 12 (a real Mac Pro), 15 and 26, checks that every Mach-O in it is x86_64 with a
  deployment target of at most 12.0, and runs MeshViewer and the examples.

> **Intent:** once proven, this replaces the GitHub-hosted `macos-15-intel` x64 leg (Intel
> runners are being retired). Until then both run and both packages are built.

## How it works

- **cmake / ninja / clang run natively as arm64.** A PATH shim puts the arm64 `cmake` and `ninja`
  first, and `CMAKE_MAKE_PROGRAM` pins that ninja: CMake's `find_program` would otherwise pick an
  x86_64 ninja from `/usr/local/bin`, which spawns clang under Rosetta and silently loses the
  speed. A translated parent makes every child translated.
- **`-D MR_PLATFORM=APPLE_x86_64`** labels the binaries: cmake is native, so
  `CMAKE_SYSTEM_PROCESSOR` stays `arm64`.
- **vcpkg** uses the triplet
  [`x64-cross-osx-meshlib`](../../thirdparty/vcpkg/triplets/x64-cross-osx-meshlib.cmake), which is
  `x64-osx-meshlib` under another name so that the caches (S3, GitHub) of the cross and the
  hosted leg stay apart. The host triplet is left equal to the target one, as on an Intel
  runner: the cmake-driven compiles still use the native clang, and the x86_64 host tools vcpkg
  builds (boost's b2, CPython's own build steps) run under Rosetta. With `arm64-osx-meshlib` as
  the host triplet CPython is configured in cross mode and fails on an undeclared `sendfile`.
  The first run builds every port from source (about 10 minutes); later runs restore them.
- **The x86_64 Homebrew at `/usr/local`** (next to the native arm64 one) provides the x86_64
  llvm-pgo keg and the GNU `make` for the bindings generation. The shim's `brew` wrapper makes
  every `brew --prefix` in the job resolve to it.
- **Rust bindings** are built with `CARGO_BUILD_TARGET=x86_64-apple-darwin` (the `target` input
  of `build-rust-bindings`). **Java bindings** are skipped: the host's arm64 JVM cannot load
  x86_64 libraries.

## What still runs under Rosetta

1. The `/usr/local` brew (its Ruby is x86_64) and the x86_64 host tools of vcpkg.
2. The bindings generation: `scripts/mrbind/generate.mk` has no target-arch flag and yields
   x86_64 modules because the x86_64 `make` and the x86_64 keg's clang run translated. A native
   `make` plus an explicit `-arch x86_64` would remove this.
3. Running the Intel output: MRTest, MRTestC2, MeshViewer, the Python tests (with vcpkg's x86_64
   `python3`), the Rust examples and tests. On this runner `DYLD_*` variables set in a step's
   `env` are stripped by System Integrity Protection; export them inside the script.

## Provisioning a runner

Run [`provision-runner.sh`](provision-runner.sh) as the CI service account (no sudo). Where root
is needed (Rosetta 2, the `/usr/local` directory skeleton owned by the account) it prints the
one-time command for an administrator and exits; re-run it afterwards. It then clones Homebrew
into `/usr/local/Homebrew` and installs the few formulae CI uses. Homebrew moved Intel macOS to
Tier 3 in 2026-09 (no new bottles, end of support planned for 2027), so keep that prefix
pinned: CI sets `HOMEBREW_NO_AUTO_UPDATE` and `HOMEBREW_NO_INSTALL_UPGRADE`; never
`brew upgrade` it by hand.

## Reproducing locally

With a vcpkg tree for the triplet installed by `scripts/build_vcpkg_thirdparty.sh`:

```bash
export VCPKG_TRIPLET=x64-cross-osx-meshlib MESHLIB_USE_VCPKG=ON VCPKG_MANIFEST_MODE=ON
export CMAKE_OSX_ARCHITECTURES=x86_64 CMAKE_MAKE_PROGRAM="$(brew --prefix)/bin/ninja"
export MESHLIB_BUILD_RELEASE=ON MESHLIB_BUILD_DEBUG=OFF
MR_CMAKE_OPTIONS="-D MR_PLATFORM=APPLE_x86_64 -D MR_CXX_STANDARD=23" ./scripts/build_source.sh
lipo -archs build/Release/bin/libMRMesh.dylib   # x86_64
```
