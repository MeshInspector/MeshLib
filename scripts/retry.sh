#!/bin/bash
set -eu

times=3
cooldown=30

usage() {
    echo "Usage: $0 [--times N] [--cooldown S] -- COMMAND [ARGS...]" >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --times)    times="${2:?--times requires a value}"; shift 2 ;;
        --cooldown) cooldown="${2:?--cooldown requires a value}"; shift 2 ;;
        --)         shift; break ;;
        *)          echo "Unknown argument: $1" >&2; usage ;;
    esac
done

[[ "$times"    =~ ^[1-9][0-9]*$ ]] || { echo "--times must be a positive integer"         >&2; exit 2; }
[[ "$cooldown" =~ ^[0-9]+$       ]] || { echo "--cooldown must be a non-negative integer" >&2; exit 2; }
[ $# -ge 1 ] || { echo "Missing command after --" >&2; usage; }

rc=0
for attempt in $(seq 1 "$times"); do
    rc=0
    # `|| rc=$?` captures the exit code without tripping `set -e`.
    "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
        exit 0
    fi
    if [ "$attempt" -lt "$times" ]; then
        echo "$(basename "$0"): attempt $attempt/$times failed (exit $rc); retrying in ${cooldown}s..." >&2
        sleep "$cooldown"
    fi
done
echo "$(basename "$0"): command failed after $times attempts (exit $rc): $*" >&2
exit "$rc"
