#!/bin/bash

# --skip-prebuilt-thirdparty: the thirdparty comes prebuilt in the Docker image
# --wheel: also mrbind and mrbind-pybind11, for the Python bindings and their shims

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

SUBMODULES=(
    thirdparty/imgui
)

SKIP_THIRDPARTY=0
WHEEL=0
for ARG in "$@"; do
    case "$ARG" in
        --skip-prebuilt-thirdparty) SKIP_THIRDPARTY=1 ;;
        --wheel) WHEEL=1 ;;
        *) echo "unknown argument: $ARG" >&2; exit 1 ;;
    esac
done

if [ $WHEEL = 1 ]; then
    SUBMODULES+=(
        thirdparty/mrbind
        thirdparty/mrbind-pybind11
    )
fi

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/.. "${SUBMODULES[@]}" || exit 1
if [ $WHEEL = 1 ]; then
    "$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/../thirdparty/mrbind deps/cppdecl || exit 1
fi
if [ $SKIP_THIRDPARTY = 0 ]; then
    "$SCRIPT_DIR"/clone_submodules_emscripten_thirdparty.sh
fi
