// Force-included via `-include` from `compiler_only_flags.txt` so that
// preprocessor macros whose values contain backslashes (most notably
// `MB_PB11_ADJUST_NAMES`) reach clang via C source rather than via the
// shell. Routing the value through `-D'"s/\\bMR:::/g"'` is fragile:
// different MSYS2 bash/make versions strip a different number of
// backslashes, leading to PCH ↔ fragment macro-mismatch errors of the
// form
//     definition of macro 'MB_PB11_ADJUST_NAMES' differs between the
//     precompiled header ('"s/\bMR:://g"') and the command line
//     ('"s/\\bMR:://g"')
// Defining the macro in a force-included header eliminates the shell
// layer entirely so the same string literal is seen on every TU.

#pragma once

#ifndef MB_PB11_ADJUST_NAMES
#define MB_PB11_ADJUST_NAMES "s/\\bMR:://g"
#endif
