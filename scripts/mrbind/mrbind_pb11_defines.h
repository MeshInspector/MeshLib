// Pulled in from the generated `<module>.combined.hpp` (see the recipe
// in `generate.mk`) so that preprocessor macros whose values are string
// literals containing backslashes (most notably `MB_PB11_ADJUST_NAMES`,
// whose value `"s/\\bMR:::/g"` carries a regex word-boundary `\b`)
// reach clang as C source rather than as a `-D` flag.
//
// Routing such values through the make+bash recipe pipeline is fragile.
// On the GitHub `windows-2022` runner with the preinstalled `C:\msys64`,
// the value reaches clang's PCH-build and per-fragment compile
// inconsistently, producing
//     error: definition of macro 'MB_PB11_ADJUST_NAMES' differs between
//     the precompiled header ('"s/\bMR:://g"') and the command line
//     ('"s/\\bMR:://g"')
//
// Pinned to the make+bash recipe pipeline by an instrumented probe — a
// `clang++` shim that logged argv (text + hex) before exec'ing the
// real clang. The PCH-build invocation actually receives the value
// with ONE backslash (`-DMB_PB11_ADJUST_NAMES="s/\bMR:://g"`), while
// the per-fragment invocation receives it with TWO
// (`-DMB_PB11_ADJUST_NAMES="s/\\bMR:://g"`) — even though make's
// `--trace` echoes identical recipe text for both. Bash's `'…'`
// quote-stripping is deterministic, so identical input → identical
// argv; the asymmetric input must come from make.
//
// Tool-version comparison between the two MSYS2 environments where
// the build works (master) vs. fails (this branch):
//
//   tool             master S3 zip                  runner C:\msys64
//   ---------------  -----------------------------  -----------------------------
//   GNU make         4.4.1-2 (MSYS2/cygwin build,   4.4.1     (mingw64 build,
//                    /usr/bin/make)                  /c/mingw64/bin/make,
//                                                    NOT pacman-managed)
//   bash             5.2.037-1                      5.3.009-1
//   coreutils        8.32-5                         8.32-5  (same)
//   msys2-runtime    3.5.4-7                        3.6.9-1
//
// Same upstream make version (4.4.1) but very different builds:
// master's make goes through cygwin's POSIX layer; the runner's make
// is a Windows-native mingw64 binary that came from the runner image's
// separate Windows-side MinGW install (the runner's MSYS2 base set
// doesn't include make, so PATH lookup falls through to that). The
// asymmetric backslash handling between PCH and fragment recipes is
// most likely a side effect of the mingw64 make's Windows-style argv
// quoting interacting with `$(if $(is_py), …)`/`$(eval)` differently
// than the cygwin make does. The `bash` major bump (5.2 → 5.3) may
// also contribute, but bash's `'…'` quoting is well-specified so the
// make-build difference is the more likely root.
//
// Defining the macro inside the PCH source eliminates the round-trip:
// clang stores and validates the same C-source spelling on both ends.
//
// Not routed via `-include`: clang requires the PCH-import `-include`
// to be the first one on the command line, and adding any other
// `-include` would push it out of first position and silently disable
// the PCH.
//
// The recipe wraps the `#include` in `#ifndef MR_PARSING_FOR_PB11_BINDINGS`
// so the mrbind parser pass — which doesn't have `scripts/mrbind/` on
// its include path and doesn't need this macro anyway — skips it.

#pragma once

#ifndef MB_PB11_ADJUST_NAMES
#define MB_PB11_ADJUST_NAMES "s/\\bMR:://g"
#endif
