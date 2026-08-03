#!/bin/bash
set -eu

times=3
cooldown=30
timeout=0
kill_after=10

usage() {
    echo "Usage: $0 [--times N] [--cooldown S] [--timeout S] -- COMMAND [ARGS...]" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --times)    times="${2:?--times requires a value}"; shift 2 ;;
        --cooldown) cooldown="${2:?--cooldown requires a value}"; shift 2 ;;
        --timeout)  timeout="${2:?--timeout requires a value}"; shift 2 ;;
        --)         shift; break ;;
        *)          echo "Unknown argument: $1" >&2; usage ;;
    esac
done

[[ "$times"    =~ ^[1-9][0-9]*$ ]] || { echo "--times must be a positive integer"         >&2; exit 2; }
[[ "$cooldown" =~ ^[0-9]+$       ]] || { echo "--cooldown must be a non-negative integer" >&2; exit 2; }
[[ "$timeout"  =~ ^[0-9]+$       ]] || { echo "--timeout must be a non-negative integer"  >&2; exit 2; }
[ $# -ge 1 ] || { echo "Missing command after --" >&2; usage; }

# Insist on GNU timeout: macOS calls it gtimeout, Windows has an unrelated System32\timeout.exe.
timeout_bin=
if [ "$timeout" -gt 0 ]; then
    for candidate in timeout gtimeout; do
        if "$candidate" --version 2>/dev/null | grep -q coreutils; then
            timeout_bin="$candidate"
            break
        fi
    done
    [ -n "$timeout_bin" ] || echo "$(basename "$0"): GNU timeout not found; running without a time limit" >&2
fi

run_attempt() {
    if [ -n "$timeout_bin" ]; then
        "$timeout_bin" --kill-after "${kill_after}s" "${timeout}s" "$@"
    else
        "$@"
    fi
}

rc=0
for attempt in $(seq 1 "$times"); do
    rc=0
    # `|| rc=$?` captures the exit code without tripping `set -e`.
    run_attempt "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
        exit 0
    fi
    if [ "$attempt" -lt "$times" ]; then
        reason="failed (exit $rc)"
        if [ -n "$timeout_bin" ]; then
            case "$rc" in
                # 137 is our SIGKILL escalation: timed out, then ignored SIGTERM.
                124) reason="timed out after ${timeout}s" ;;
                137) reason="timed out after ${timeout}s, SIGKILLed ${kill_after}s later" ;;
            esac
        fi
        echo "$(basename "$0"): attempt $attempt/$times $reason; retrying in ${cooldown}s..." >&2
        sleep "$cooldown"
    fi
done
echo "$(basename "$0"): command failed after $times attempts (exit $rc): $*" >&2
exit "$rc"
