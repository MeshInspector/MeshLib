// Pulled in from the generated `<module>.combined.hpp` (see the recipe
// in `generate.mk`) so that preprocessor macros whose values are string
// literals containing backslashes (most notably `MB_PB11_ADJUST_NAMES`,
// whose value `"s/\\bMR:::/g"` carries a regex word-boundary `\b`)
// reach clang as C source rather than as a `-D` flag.
//
// Routing such values through the make+bash recipe pipeline is fragile:
// on some MSYS2 environments (confirmed: the runner's preinstalled
// `C:\msys64` with bash 5.3.9 / GNU make 4.x) the value reaches clang's
// PCH-build and per-fragment compile inconsistently, producing
//     error: definition of macro 'MB_PB11_ADJUST_NAMES' differs between
//     the precompiled header ('"s/\bMR:://g"') and the command line
//     ('"s/\\bMR:://g"')
//
// Pinned to GNU make by an instrumented probe: a `clang++` shim that
// logged argv (text + hex) before exec'ing the real clang showed the
// PCH-build invocation actually receives the value with ONE backslash
// (`-DMB_PB11_ADJUST_NAMES="s/\bMR:://g"`), while the per-fragment
// invocation receives it with TWO backslashes
// (`-DMB_PB11_ADJUST_NAMES="s/\\bMR:://g"`) — even though make's
// `--trace` echoes identical recipe text for both. Bash's `'…'`
// quote-stripping is deterministic, so identical input → identical
// argv; the asymmetric input must come from make. The PCH recipe is
// nested inside `$(if $(is_py), …)` in a `define module_snippet_build_py`
// block that goes through `$(eval)`, while the fragment recipe is a
// top-level rule in the same block — the extra round of make-variable
// expansion appears to consume one backslash on the PCH path that the
// fragment path keeps.
//
// Master with its frozen S3 MSYS2 snapshot (older bash + older make)
// doesn't trip this — the older make evidently constructs recipe text
// without the extra backslash strip, so PCH and fragment invocations
// receive identical bytes and clang's PCH validator is happy.
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
