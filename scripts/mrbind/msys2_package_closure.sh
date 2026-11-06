#!/bin/bash
# Usage: msys2_package_closure.sh SUFFIX PACKAGE...
# Prints the lines of `msys2_package_hashes<SUFFIX>.txt` for the given packages and everything they depend on,
#   read from the downloaded package files. Packages are named without the `mingw-w64-clang-<arch>-` prefix (`clang`, `make`).
# Dependencies outside of the list (`bash` and the rest of the MSYS2 base) are assumed to be installed already.
set -euo pipefail

cd "$(dirname "$BASH_SOURCE")"

SUFFIX="$1"
shift
HASH_FILE="msys2_package_hashes${SUFFIX}.txt"

base_name() { local n="${1%%[<>=]*}"; echo "${n#mingw-w64-clang-*-}"; }

declare -A LINE_OF DEPS_OF PROVIDER
mapfile -t LINES < <(tr -d '\r' <"${HASH_FILE}")
FILES=()
for line in "${LINES[@]}"; do
    FILES+=("${line#*" *"}")
done

# One `pacman` for all the files: `Name` and `Depends On` of each, in the order of `FILES`.
i=-1
while IFS= read -r info; do
    case $info in
        "Name "*)
            i=$((i + 1))
            name="$(base_name "${info#*: }")"
            LINE_OF[$name]="${LINES[$i]}"
            ;;
        "Depends On "*)
            deps="${info#*: }"
            [[ $deps == None ]] && deps=
            DEPS_OF[$name]="$deps"
            ;;
        "Provides "*)
            # E.g. `libc++` provides `cc-libs`, which `clang` and `lld` depend on.
            for p in ${info#*: }; do
                [[ $p == None ]] || PROVIDER[$(base_name "$p")]="$name"
            done
            ;;
    esac
done < <(pacman -Qip "${FILES[@]}" | grep -E '^(Name|Depends On|Provides) ')
[[ $((i + 1)) == ${#FILES[@]} ]] || { echo "Read $((i + 1)) of ${#FILES[@]} packages from ${HASH_FILE}" >&2; exit 1; }

declare -A SEEN
QUEUE=("$@")
while ((${#QUEUE[@]})); do
    name="${QUEUE[0]}"
    QUEUE=("${QUEUE[@]:1}")
    [[ -z ${LINE_OF[$name]:-} && -n ${PROVIDER[$name]:-} ]] && name="${PROVIDER[$name]}"
    [[ -n ${SEEN[$name]:-} || -z ${LINE_OF[$name]:-} ]] && continue
    SEEN[$name]=1
    for dep in ${DEPS_OF[$name]}; do
        QUEUE+=("$(base_name "$dep")")
    done
done

for name in "$@"; do
    [[ -n ${LINE_OF[$name]:-} ]] || { echo "No package \`$name\` in ${HASH_FILE}" >&2; exit 1; }
done
declare -A KEEP
for name in "${!SEEN[@]}"; do
    KEEP[${LINE_OF[$name]}]=1
done
for line in "${LINES[@]}"; do
    [[ -n ${KEEP[$line]:-} ]] && echo "$line"
done
exit 0
