#!/bin/bash
# CI step of the x64-cross leg (build-test-macos.yml): prepares the job for
# building x86_64 on the arm64 runner with the native toolchain, see README.md.
# Must run before the compiler setup and any other `brew --prefix`.
#
# Puts a shim directory first on the job's PATH ($GITHUB_PATH) with
#   cmake, ninja - the native arm64 ones, so that clang is not run under Rosetta;
#   brew         - the x86_64 Homebrew at /usr/local, which has the x86_64 llvm-pgo
#                  keg and the GNU make that drives the bindings generation;
# and exports CMAKE_OSX_ARCHITECTURES and CMAKE_MAKE_PROGRAM ($GITHUB_ENV), which
# scripts/build_source.sh forwards to cmake.
#
# VCPKG_DEFAULT_HOST_TRIPLET is deliberately left alone: the host triplet stays
# equal to the target one (README.md, "How it works").
set -euo pipefail

fail() {
  echo "::error::x86_64 cross setup: $1"
  exit 1
}

X64_BREW=/usr/local/bin/brew
SHIM="$RUNNER_TEMP/x86_64-cross-shim"

# The native arm64 Homebrew prefix varies across the self-hosted fleet; find it by
# its arm64 cmake and ninja. The x86_64 copies under /usr/local must not be used.
ARM_BREW=""
for prefix in /opt/homebrew "$HOME/.homebrew"; do
  if file -b "$prefix/bin/cmake" 2>/dev/null | grep -q arm64 &&
     file -b "$prefix/bin/ninja" 2>/dev/null | grep -q arm64; then
    ARM_BREW="$prefix"
    break
  fi
done
[ -n "$ARM_BREW" ] || fail "no arm64 Homebrew cmake and ninja found"
[ -x "$X64_BREW" ] || fail "missing $X64_BREW"

mkdir -p "$SHIM"
ln -sf "$ARM_BREW/bin/cmake" "$SHIM/cmake"
ln -sf "$ARM_BREW/bin/ninja" "$SHIM/ninja"
# A wrapper, not a symlink: a symlink to a symlink breaks Homebrew's self-location.
printf '#!/bin/bash\nexec %s "$@"\n' "$X64_BREW" > "$SHIM/brew"
chmod +x "$SHIM/brew"

echo "$SHIM" >> "$GITHUB_PATH"
{
  echo "CMAKE_OSX_ARCHITECTURES=x86_64"
  echo "CMAKE_MAKE_PROGRAM=$ARM_BREW/bin/ninja"
} >> "$GITHUB_ENV"
