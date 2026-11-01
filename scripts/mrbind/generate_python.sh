#!/bin/bash
# Generates the source of a Python module over the C bindings with `mrbind_gen_python`, see `generate.mk`.
# Usage: generate_python.sh <mrbind_gen_python> <interop_desc.json of the C bindings> <output dir> <mrmeshpy|mrcudapy>
# The names and docs follow what the pybind11 bindings had, so the flags below spell the types like `typeid()` on this platform.

set -euo pipefail

GEN_PYTHON="$1"
INPUT_JSON="$2"
OUTPUT_DIR="$3"
MODULE="$4"

case "$(uname -s)" in
    Linux*) OS=linux ;;
    Darwin*) OS=macos ;;
    *) OS=windows ;;
esac

SCRIPT_DIR="$(dirname "$BASH_SOURCE")"

# The paths for the native Windows generator.
native() { if [[ $OS == windows ]]; then cygpath -m "$1"; else echo "$1"; fi; }

# The pybind11-only ignores of the MeshLib headers, which the C bindings don't see.
GEN_FLAGS=(
    --replace-namespace MR ::
    --adjust-name '\bMR::' ''
    --hide-field id_
    --hide-field mask_
    # `begin()`/`end()` are function templates for those.
    --iterate-by-index '^MR::Vector[234](b|i|i64|f|d|<)'
    --ignore-for-python '^MR::SetBitIteratorT<'
    --ignore-for-python 'SetBitIterator$'
    --ignore-for-python '^MR::ConstChildren::(Iterator|begin|end)$'
    --ignore-for-python '^MR::MeshBuilder::inspectVertNeighbourhood\('
    --ignore-for-python '^MR::(begin|end)\(.*BitSet'
    # The C bindings merge `std::expected` and `tl::expected`, which is what `typeid()` shows.
    --adjust-typeid-name '(^|[^:A-Za-z0-9_])expected<' '$1tl::expected<'
)
# The types are named after the demangled `typeid()`, which spells the integer typedefs as their underlying types; so do the docs and most aliases.
case $OS in
    linux) INTS=('size_t|uint64_t' 'unsigned long' 'ptrdiff_t|int64_t' 'long') ;;
    macos) INTS=('size_t' 'unsigned long' 'uint64_t' 'unsigned long long' 'ptrdiff_t' 'long' 'int64_t' 'long long') ;;
    windows) INTS=('size_t|uint64_t' 'unsigned long long' 'ptrdiff_t|int64_t' 'long long') ;;
esac
for ((i = 0; i < ${#INTS[@]}; i += 2)); do
    GEN_FLAGS+=(--adjust-name "\\b(std::)?(${INTS[i]})\\b" "${INTS[i + 1]}" --adjust-doc-name "\\b(std::)?(${INTS[i]})\\b" "${INTS[i + 1]}")
done
# The aliases that keep `size_t` (and `uint64_t`) as written.
SIZE_T="${INTS[1]}"
GEN_FLAGS+=(
    --adjust-alias-name "^HashMap<${SIZE_T}, ${SIZE_T}>\$" 'HashMap<size_t, size_t>'
    --adjust-alias-name "^std::vector<${SIZE_T}>\$" 'std::vector<size_t>'
    --adjust-alias-name "^std::vector<std::pair<${SIZE_T}, ${SIZE_T}>>\$" 'std::vector<std::pair<size_t, size_t>>'
    --adjust-alias-name "\\b${SIZE_T}\\(" 'size_t('
)
if [[ $OS != macos ]]; then
    GEN_FLAGS+=(--adjust-alias-name "\\bphmap::Hash<${SIZE_T}>" 'phmap::Hash<uint64_t>')
fi
# The description comes from Linux, where the `size_t` literals are `0ul`.
if [[ $OS == windows ]]; then
    GEN_FLAGS+=(--adjust-alias-name '\b([0-9]+)ul\b' '$1ull')
fi
# `typeid()` expands the alias templates and `int32_t`, and spells the default template arguments.
GEN_FLAGS+=(
    --adjust-typeid-name '\bMR::VoxelValueGetter<([^<>]*)>' 'std::function<$1(MR::Vector3<int> const &)>'
    --adjust-typeid-name '\b(std::)?int32_t\b' 'int'
)
if [[ $OS != windows ]]; then
    # How the Itanium demangler spells these: the size of a nested `std::array` and the other `size_t` template arguments as `3ul`.
    GEN_FLAGS+=(
        --adjust-typeid-name '([<,] ?std::array<[^<>]*(?:<[^<>]*(?:<[^<>]*>)?[^<>]*>)?[^<>]*), ([0-9]+)>' '$1, $2ul>'
        --adjust-typeid-name '((?:^|[<,] ?)MR::Polynomial<[^<>,]+, )([0-9]+)>' '$1$2ul>'
        --adjust-typeid-name '([<,] ?MR::Vector<[^<>,]+)>' '$1, unsigned long>'
    )
    # libc++ has `std::filesystem` in an inline namespace.
    if [[ $OS == macos ]]; then
        GEN_FLAGS+=(--adjust-typeid-name '\bstd::filesystem::path\b' 'std::__fs::filesystem::path')
    fi
else
    # How MSVC spells these: the calling convention of the function types.
    GEN_FLAGS+=(
        --adjust-typeid-name '([<,] ?MR::Vector<[^<>,]+)>' '$1, unsigned long long>'
        --adjust-typeid-name '([<,] ?std::function<[^()<>]*)\(' '$1 __cdecl('
    )
fi

case $MODULE in
    mrmeshpy)
        GEN_FLAGS+=(--ignore-c-file '^MRCCuda/' --extra-python "$(native "$SCRIPT_DIR/python_helpers.py")")
        while read -r alias target; do
            GEN_FLAGS+=(--alias "$alias" "$target")
        done < <(grep -v '^#' "$SCRIPT_DIR/python_aliases.txt" | tr -d '\r')
        ;;
    mrcudapy)
        # Only the Cuda headers, with the classes of `mrmeshpy`.
        GEN_FLAGS+=(--ignore-c-file '^(?!MRCCuda/)' --depends-on meshlib.mrmeshpy)
        ;;
    *)
        echo "Unknown module: $MODULE" >&2
        exit 1
        ;;
esac

# The specific namespace replacement first.
if [[ $MODULE == mrcudapy ]]; then
    GEN_FLAGS=(--replace-namespace MR::Cuda :: "${GEN_FLAGS[@]}")
fi

"$GEN_PYTHON" --input-json "$(native "$INPUT_JSON")" --output-dir "$(native "$OUTPUT_DIR")" --clean-output-dir \
    --module-name "$MODULE" --module-qualname "meshlib.$MODULE" "${GEN_FLAGS[@]}" --verbose 2> "$OUTPUT_DIR.log" || { tail -20 "$OUTPUT_DIR.log" >&2; exit 1; }
grep '^Generated' "$OUTPUT_DIR.log"
