#!/bin/bash
# Diagnostic shim for bash, used as `make SHELL=...`. Logs every argv
# element (text + hex) to `$DIAG_LOG_DIR/shell-recipes.log`, then
# exec's the real bash. Captures the recipe text make passes to bash
# (pre-quote-stripping) so we can compare it against what clang
# actually receives via the clang shim (post-quote-stripping).

LOG_DIR="${DIAG_LOG_DIR:-/c/diag}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/shell-recipes.log"

{
  printf '====\nBASH_INVOCATION pid=%d ppid=%d argc=%d\n' "$$" "$PPID" "$#"
  i=0
  for a in "$@"; do
    printf 'ARG[%d]_TEXT: %s\n' "$i" "$a"
    printf 'ARG[%d]_HEX:  ' "$i"
    printf '%s' "$a" | od -An -tx1 | tr -d ' \n'
    printf '\n'
    i=$((i+1))
  done
} >> "$LOG"

exec /usr/bin/bash "$@"
