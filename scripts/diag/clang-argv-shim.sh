#!/bin/bash
# Diagnostic shim for clang++. Logs every argv element (text + hex) to
# `$DIAG_LOG_DIR/clang-argv.log`, then exec's the real clang++.
# Used to capture exactly what bash passes to clang during the
# MB_PB11_ADJUST_NAMES PCH ↔ fragment macro-mismatch investigation.

LOG_DIR="${DIAG_LOG_DIR:-/c/diag}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/clang-argv.log"

{
  printf '====\nINVOCATION pid=%d ppid=%d argc=%d cwd=%s\n' "$$" "$PPID" "$#" "$(pwd)"
  i=0
  for a in "$@"; do
    printf 'ARGV[%d]_TEXT: %s\n' "$i" "$a"
    printf 'ARGV[%d]_HEX:  ' "$i"
    printf '%s' "$a" | od -An -tx1 | tr -d ' \n'
    printf '\n'
    i=$((i+1))
  done
} >> "$LOG"

exec /c/msys64/clang64/bin/clang++.exe "$@"
