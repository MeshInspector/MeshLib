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
)

# `aliases.cpp` has the custom aliases of the pybind11 module.
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

# `mrmeshnumpy` is a pybind11 module over the pybind11 `mrmeshpy`, so it's stubbed out for now.
cat > "$OUT_DIR/pkg/meshlib/__init__.py" <<'EOF'
EOF
cat > "$OUT_DIR/pkg/meshlib/mrmeshnumpy.py" <<'EOF'
def __getattr__(name):
    raise NotImplementedError("mrmeshnumpy is not available over the C bindings yet")
EOF

DUMP="$MRBIND_DIR/scripts/python_api_dump.py"
"$PYTHON" "$DUMP" dump meshlib.mrmeshpy --path "$BIN_DIR" --strip meshlib.mrmeshpy. -o "$OUT_DIR/api_pybind11.json"
"$PYTHON" "$DUMP" dump meshlib.mrmeshpy --path "$OUT_DIR/pkg" --strip meshlib.mrmeshpy. -o "$OUT_DIR/api_over_c.json"
"$PYTHON" "$DUMP" diff "$OUT_DIR/api_pybind11.json" "$OUT_DIR/api_over_c.json" --limit 1000000 > "$OUT_DIR/api_diff.txt" || true
head -1 "$OUT_DIR/api_diff.txt"
grep '^== ' "$OUT_DIR/api_diff.txt" || true
