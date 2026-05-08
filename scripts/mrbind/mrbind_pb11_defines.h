// Pulled in from the generated `<module>.combined.hpp` (see the recipe
// in `generate.mk`) so that preprocessor macros whose values contain
// backslashes (most notably `MB_PB11_ADJUST_NAMES`) reach clang via C
// source rather than via the shell. Routing the value through
// `-D'"s/\\bMR:://g"'` is fragile: different MSYS2 bash/make versions
// strip a different number of backslashes, leading to PCH ↔ fragment
// macro-mismatch errors of the form
//     definition of macro 'MB_PB11_ADJUST_NAMES' differs between the
//     precompiled header ('"s/\bMR:://g"') and the command line
//     ('"s/\\bMR:://g"')
// Defining the macro inside the PCH source eliminates the shell layer
// entirely so the same string literal is seen on every TU.
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
