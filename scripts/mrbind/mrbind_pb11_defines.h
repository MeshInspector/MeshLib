// Pulled in from the generated `<module>.combined.hpp` (see the recipe
// in `generate.mk`) so that preprocessor macros whose values are string
// literals containing backslashes (most notably `MB_PB11_ADJUST_NAMES`,
// whose value `"s/\\bMR:::/g"` carries a regex word-boundary `\b`)
// reach clang as C source rather than as a `-D` flag.
//
// Defining via `-D` triggers a clang quirk: clang's PCH validation
// re-renders the `-D` macro value when comparing the PCH-stored form
// against the new TU's command line, and the re-rendered form drops a
// backslash, so identical bytes-on-the-wire produce a textual mismatch:
//     error: definition of macro 'MB_PB11_ADJUST_NAMES' differs between
//     the precompiled header ('"s/\bMR:://g"') and the command line
//     ('"s/\\bMR:://g"')
// (Verified by hex-dumping the build log: bash receives bytewise
// identical `-D...='"s/\\bMR:::/g"'` on both PCH-build and per-fragment
// compile, so neither make nor bash is the culprit — the asymmetry is
// inside clang's PCH stringifier for `-D` macros.)
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
