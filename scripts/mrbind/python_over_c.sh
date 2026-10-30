#!/bin/bash
# Experimental: generates `mrmeshpy` over the C bindings with `mrbind_gen_python` instead of pybind11,
#   and compares its API with the pybind11 module.
# Usage: python_over_c.sh <the build's bin directory, with libMeshLibC2 and the pybind11 `meshlib` package>
# Run after `make -f scripts/mrbind/generate.mk TARGET=c` (for `interop_desc.json`) and the build of `MeshLibC2`.
# The output goes to `build/python_over_c`: `pkg/meshlib/mrmeshpy.so`, and the API dumps and their diff.

set -euxo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"
BIN_DIR="$(realpath "$1")"
MRBIND_DIR="$ROOT_DIR/thirdparty/mrbind"
OUT_DIR="$ROOT_DIR/build/python_over_c"
PYTHON="${PYTHON:-python3}"
# The same compiler as `CXX_FOR_BINDINGS` in `generate.mk`.
if [[ -n "${LLVM_PREFIX:-}" ]]; then
    CXX="$LLVM_PREFIX/bin/clang++"
else
    CXX="clang++-$(cat "$SCRIPT_DIR/clang_version.txt" | xargs)"
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/pkg/meshlib"

# Like `MB_PB11_STRIPPED_NAMESPACES`, `MB_PB11_ADJUST_NAMES` and the pybind11-only ignores in the MeshLib headers.
GEN_FLAGS=(
    --replace-namespace MR ::
    --adjust-name '\bMR::' ''
    --hide-field id_
    # `mrcudapy` is a separate module.
    --ignore-c-file '^MRCCuda/'
    # `begin()`/`end()` are function templates for those.
    --iterate-by-index '^MR::Vector[234](b|i|i64|f|d|<)'
    --ignore-for-python '^MR::SetBitIteratorT<'
    --ignore-for-python 'SetBitIterator$'
    --ignore-for-python '^MR::ConstChildren::(Iterator|begin|end)$'
    --ignore-for-python '^MR::MeshBuilder::inspectVertNeighbourhood\('
    --ignore-for-python '^MR::(begin|end)\(.*BitSet'
    # The pybind11 target names the types after the demangled `typeid()`, spelled like this on Linux.
    --adjust-name '\b(std::)?(size_t|uint64_t)\b' 'unsigned long'
    --adjust-name '\b(std::)?(ptrdiff_t|int64_t)\b' 'long'
    # The pybind11 build is parsed without `size_t` support, so its docs spell the underlying types.
    --adjust-doc-name '\b(std::)?(size_t|uint64_t)\b' 'unsigned long'
    --adjust-doc-name '\b(std::)?(ptrdiff_t|int64_t)\b' 'long'
    # The C bindings merge `std::expected` and `tl::expected`, which is what `typeid()` shows.
    --adjust-typeid-name '(^|[^:A-Za-z0-9_])expected<' '$1tl::expected<'
    # How the Linux demangler spells these: the size of a nested `std::array` as `3ul`, and `int32_t` expanded.
    --adjust-typeid-name '([<,] ?std::array<[^<>]*(?:<[^<>]*(?:<[^<>]*>)?[^<>]*>)?[^<>]*), ([0-9]+)>' '$1, $2ul>'
    --adjust-typeid-name '\b(std::)?int32_t\b' 'int'
)

# `aliases.cpp` has the custom aliases of the pybind11 module; its `helpers.cpp` functions are in `helpers_over_c.py`.
ALIAS_FLAGS=()
while read -r alias target; do
    ALIAS_FLAGS+=(--alias "$alias" "$target")
done < <(sed -nE 's/^\s*MR_ALIAS\(\s*([A-Za-z0-9_.]+)\s*,\s*([A-Za-z0-9_.]+)\s*\);.*/\1 \2/p' "$SCRIPT_DIR/aliases.cpp")

"$MRBIND_DIR/build/mrbind_gen_python" \
    --input-json "$ROOT_DIR/source/MeshLibC2/temp/interop_desc.json" \
    --output-dir "$OUT_DIR/src" \
    --clean-output-dir \
    --module-name mrmeshpy \
    --module-qualname meshlib.mrmeshpy \
    "${GEN_FLAGS[@]}" \
    "${ALIAS_FLAGS[@]}" \
    --extra-python "$SCRIPT_DIR/helpers_over_c.py" \
    --verbose 2> "$OUT_DIR/gen.log"
grep '^Generated' "$OUT_DIR/gen.log" || true

# The limited API, so one module serves every Python version.
time "$CXX" \
    "$OUT_DIR/src/mrmeshpy.cpp" \
    -o "$OUT_DIR/pkg/meshlib/mrmeshpy.so" \
    -shared -fPIC -std=c++20 -O1 -w \
    -DPy_LIMITED_API=0x03080000 \
    -I"$OUT_DIR/src" \
    -I"$ROOT_DIR/source/MeshLibC2/include" \
    $("$PYTHON-config" --includes) \
    -L"$BIN_DIR" -lMeshLibC2 \
    -Wl,-rpath,"$BIN_DIR"

# `mrmeshnumpy` in Python over this `mrmeshpy`; the module only re-exports the public functions.
cat > "$OUT_DIR/pkg/meshlib/__init__.py" <<'EOF'
EOF
cp "$SCRIPT_DIR/mrmeshnumpy_over_c.py" "$OUT_DIR/pkg/meshlib/_mrmeshnumpy.py"
cat > "$OUT_DIR/pkg/meshlib/mrmeshnumpy.py" <<'EOF'
from ._mrmeshnumpy import *  # noqa: F401,F403
EOF

# `mrviewerpy` over this `mrmeshpy`, if built: the `mrviewerpy_over_c` target (`MESHLIB_BUILD_PYTHON_OVER_C`).
MODULES=(mrmeshpy mrmeshnumpy)
if compgen -G "$BIN_DIR/python_over_c/meshlib/mrviewerpy*" > /dev/null; then
    cp "$BIN_DIR"/python_over_c/meshlib/mrviewerpy* "$OUT_DIR/pkg/meshlib/"
    # MeshLib's pybind11 loads these from the module's directory.
    cp "$BIN_DIR"/meshlib/*pybind11nonlimitedapi* "$OUT_DIR/pkg/meshlib/" || true
    MODULES+=(mrviewerpy)
fi

DUMP="$MRBIND_DIR/scripts/python_api_dump.py"
for MODULE in "${MODULES[@]}"; do
    "$PYTHON" "$DUMP" dump meshlib.$MODULE --path "$BIN_DIR" --strip meshlib.$MODULE. -o "$OUT_DIR/api_${MODULE}_pybind11.json"
    "$PYTHON" "$DUMP" dump meshlib.$MODULE --path "$OUT_DIR/pkg" --strip meshlib.$MODULE. -o "$OUT_DIR/api_${MODULE}_over_c.json" || { echo "::warning::Can't import the $MODULE over the C bindings"; continue; }
    "$PYTHON" "$DUMP" diff "$OUT_DIR/api_${MODULE}_pybind11.json" "$OUT_DIR/api_${MODULE}_over_c.json" --limit 1000000 > "$OUT_DIR/api_${MODULE}_diff.txt" || true
    head -1 "$OUT_DIR/api_${MODULE}_diff.txt"
    grep '^== ' "$OUT_DIR/api_${MODULE}_diff.txt" || true
done

# The Python tests, like `run_python_test_script.py` runs them.
(cd "$ROOT_DIR/test_python" && MeshLibPyModulesPath="$OUT_DIR/pkg" "$PYTHON" -m pytest -q -p no:cacheprovider -rA --tb=line --basetemp="$OUT_DIR/pytest_temp" > "$OUT_DIR/tests.txt" 2>&1) || true
tail -1 "$OUT_DIR/tests.txt"
