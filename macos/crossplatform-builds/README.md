# macOS cross-platform builds — native (non-Rosetta) x86_64 on Apple Silicon

Build the **Intel (`x86_64`)** macOS target of MeshLib on **Apple Silicon (`arm64`)** using a
**native arm64 toolchain that cross-targets x86_64** — the compiler runs natively (no Rosetta
translation of the build tools, so compiles are fast) and emits x86_64 code via `-arch x86_64`,
linking the x86_64 libraries from the `/usr/local` Homebrew.

> This is the alternative to the "run everything under Rosetta" approach: there the whole
> toolchain (cmake/ninja/clang) runs as x86_64 under Rosetta, which is simpler but slower to
> compile. Here the toolchain stays native arm64 and only the *output* is x86_64.

## How it works

- **cmake / ninja / clang run natively as arm64** (fast). `CMAKE_OSX_ARCHITECTURES=x86_64`
  makes AppleClang emit x86_64 objects; the resulting binaries run on Intel Macs, or on this
  host under Rosetta.
- The x86_64 dependencies come from the **x86_64 Homebrew at `/usr/local`** (coexists with the
  native arm64 Homebrew at `/opt/homebrew`). [`ConfigureHomebrew.cmake`](../../cmake/Modules/ConfigureHomebrew.cmake)
  now honors an explicit `-D HOMEBREW_PREFIX=/usr/local` instead of always calling `brew --prefix`.

## Critical gotchas (why the naive attempt silently falls back to Rosetta)

1. **Force the arm64 ninja.** CMake's `find_program` searches `/usr/local/bin` *by default*, even
   when it isn't on `PATH`, so it picks up the **x86_64** ninja from the Intel Homebrew — and an
   x86_64 ninja spawns **x86_64 clang under Rosetta**, silently defeating the native build. Pass
   `-D CMAKE_MAKE_PROGRAM=/opt/homebrew/bin/ninja`. (Verify with `vmmap <clang-pid> | grep "Code Type"`
   → must say `ARM64`, not `X86-64 (Translated)`.)
2. **Point find_package at `/usr/local`** with `-D CMAKE_PREFIX_PATH=/usr/local` so packages like
   `Python`, OpenSSL, etc. resolve their x86_64 copies (not the arm64 `/opt/homebrew` ones).
3. **x86_64 Python vs native cmake PATH tension.** Native cmake/ninja want `/opt/homebrew` first
   on `PATH`, but Python must be the x86_64 one from `/usr/local`. Resolve with a small shim dir
   on `PATH` that maps `cmake`/`ninja` → `/opt/homebrew` (arm64) and `python3.10`/`python3.10-config`
   → `/usr/local` (x86_64).
4. **`CMAKE_SYSTEM_PROCESSOR` stays `arm64`** (it reflects the *host*, since the cmake process is
   native). This is harmless for MeshLib's own code (its SIMD is gated on target macros
   `__x86_64__`/`__aarch64__`), and the heavy SIMD dependencies (OpenVDB, TBB, blosc) come from
   x86_64 Homebrew *binaries* — not built from source here. Cosmetic side effect: `MR_PLATFORM`
   is labelled `APPLE_arm64` for an x86_64 build.
5. Configure-time feature probes (`try_run` / `find_package(Python)` running the interpreter)
   execute x86_64 test binaries, which the OS runs via Rosetta *transparently*. The bulk
   compilation is native; only these brief configure probes touch Rosetta. Going fully
   Rosetta-free would require a CMake toolchain file with `CMAKE_CROSSCOMPILING` + pre-seeded
   `try_run` results.

## What changed (vs upstream)

| Change | File |
|---|---|
| Honor `-D HOMEBREW_PREFIX=<prefix>` override (falls back to `brew --prefix`) | [`cmake/Modules/ConfigureHomebrew.cmake`](../../cmake/Modules/ConfigureHomebrew.cmake) |
| Honor a caller-set `NPROC` to cap build parallelism | [`scripts/build_source.sh`](../../scripts/build_source.sh) |

Everything else is driven by CMake `-D` flags at invocation (below), so no other source changes
are required.

## Recipe (Apple Silicon → native x86_64 cross build)

Prerequisites: Rosetta 2 (only for *running* the resulting x86_64 binaries / configure probes) and
an x86_64 Homebrew bootstrapped at `/usr/local` with the `requirements/macos.txt` formulae.

```bash
# shim: native arm64 cmake/ninja + x86_64 python on PATH
SHIM=/tmp/mlnative_bin; mkdir -p "$SHIM"
ln -sf /opt/homebrew/bin/cmake        "$SHIM/cmake"
ln -sf /opt/homebrew/bin/ninja        "$SHIM/ninja"
ln -sf /usr/local/bin/python3.10      "$SHIM/python3.10"
ln -sf /usr/local/bin/python3.10-config "$SHIM/python3.10-config"

env -i HOME="$HOME" \
  PATH="$SHIM:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/bin:/bin:/usr/sbin:/sbin" \
  NPROC=5 MESHLIB_BUILD_RELEASE=ON MESHLIB_BUILD_DEBUG=OFF \
  CMAKE_C_COMPILER=/usr/bin/clang CMAKE_CXX_COMPILER=/usr/bin/clang++ \
  MR_CMAKE_OPTIONS="\
    -D CMAKE_MAKE_PROGRAM=/opt/homebrew/bin/ninja \
    -D HOMEBREW_PREFIX=/usr/local \
    -D CMAKE_PREFIX_PATH=/usr/local \
    -D CMAKE_OSX_ARCHITECTURES=x86_64 \
    -D MR_CXX_STANDARD=23 -D MR_PCH_USE_EXTRA_HEADERS=ON" \
  bash ./scripts/build_source.sh
```

The thirdparty-from-source libraries build the same way (native arm64 tools + the same `-D` flags);
their x86_64 output is bit-for-bit equivalent regardless of whether they were built natively or
under Rosetta.

## Validation

Validated on an Apple **M4** (macOS 15.7): MeshLib core built with a **native arm64 clang**
(confirmed `Code Type: ARM64` for the live compiler processes) cross-targeting x86_64; all
binaries are x86_64 and link the `/usr/local` x86_64 Homebrew dylibs; **294/294 unit tests pass**
(the x86_64 test binary runs under Rosetta).

## Native vs Rosetta — which to use

| | Rosetta (`…-rosetta` branch) | Native (this branch) |
|---|---|---|
| Toolchain | x86_64 under Rosetta | native arm64 |
| Compile speed | slower (translated clang) | **faster (native clang)** |
| CMake setup | just PATH + a couple env vars | more `-D` flags; ninja/find_program gotchas |
| `CMAKE_SYSTEM_PROCESSOR` | `x86_64` (SIMD paths correct for source builds) | `arm64` (cosmetic `MR_PLATFORM` mislabel) |
| Robustness | higher (fewer moving parts) | needs care (silent Rosetta fallback if ninja wrong) |

Native wins on build speed; Rosetta wins on simplicity/robustness. Pick per priority.
