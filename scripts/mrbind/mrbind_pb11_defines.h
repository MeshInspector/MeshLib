// Pulled in from the generated `<module>.combined.hpp` (see the recipe
// in `generate.mk`) so that preprocessor macros whose values are string
// literals containing backslashes (most notably `MB_PB11_ADJUST_NAMES`,
// whose value `"s/\\bMR:::/g"` carries a regex word-boundary `\b`)
// reach clang as C source rather than as a `-D` flag.
//
// Routing such values through the make+bash recipe pipeline is fragile:
// on some MSYS2 environments (confirmed: the runner's preinstalled
// `C:\msys64`) the value reaches clang's PCH-build and per-fragment
// compile inconsistently, producing
//     error: definition of macro 'MB_PB11_ADJUST_NAMES' differs between
//     the precompiled header ('"s/\bMR:://g"') and the command line
//     ('"s/\\bMR:://g"')
// The same clang 18.1.8 + same `-D` flag works correctly on master's
// frozen S3 MSYS2 snapshot, so the trigger is something in newer
// bash/make/coreutils — the exact tool wasn't pinpointed (make's
// `--trace` echoes identical bytes from both recipes, but those bytes
// are evidently processed asymmetrically downstream before reaching
// clang).
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
