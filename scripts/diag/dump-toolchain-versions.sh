#!/bin/bash
# Dump versions and binary paths of the toolchain components that are
# implicated in the MB_PB11_ADJUST_NAMES PCH/fragment macro-mismatch
# investigation. Run from inside an MSYS2 -clang64 -full-path shell so
# both Windows-side and MSYS2-side tools resolve the same way the
# actual mrbind build does.
set -uo pipefail

print_section() {
  printf '\n== %s ==\n' "$1"
}

print_tool() {
  local cmd=$1
  printf '  which: '; which "$cmd" 2>&1 || true
  printf '  ver:   '; "$cmd" --version 2>&1 | head -1 || true
}

print_pacman() {
  local pkg=$1
  printf '  %s: ' "$pkg"
  if out=$(pacman -Qi "$pkg" 2>/dev/null); then
    printf '%s\n' "$out" | awk -F': ' '/^(Name|Version)/{printf "%s=%s ", $1, $2} END{print ""}'
  else
    printf '(not from pacman)\n'
  fi
}

print_section bash
print_tool bash
print_section make
print_tool make
print_section clang++
print_tool clang++
print_section uname
uname -a

print_section "pacman -Qi (runner's MSYS2 base packages)"
for p in make bash coreutils msys2-runtime ; do
  print_pacman "$p"
done
