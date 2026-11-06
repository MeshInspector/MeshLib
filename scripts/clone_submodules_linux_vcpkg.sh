#!/bin/bash

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

SUBMODULES=(
    thirdparty/imgui
    thirdparty/mrbind
)

"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/.. "${SUBMODULES[@]}"
"$SCRIPT_DIR"/checkout_submodules.sh "$SCRIPT_DIR"/../thirdparty/mrbind deps/cppdecl
