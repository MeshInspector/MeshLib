#!/bin/bash
# Experimental: generates `mrmeshpy` over the C bindings with `mrbind_gen_python` instead of pybind11,
#   and compares its API with the pybind11 module.
# Usage: python_over_c.sh <the build's bin directory, with MeshLibC2 and the pybind11 `meshlib` package>
# Run after the build of `MeshLibC2` (whose `temp/interop_desc.json` comes with the generated C bindings).
# Linux and macOS: clang++ (`LLVM_PREFIX`, else `clang++-<clang_version.txt>`).
# Windows: Git Bash under a VS developer prompt, with clang-cl (`CLANG_CL`, else the one of the MSYS2 `MSYS2_ENV`).
# The output goes to `build/python_over_c`: `pkg/meshlib/` with the modules, and the API dumps and their diff.

set -euxo pipefail

case "$(uname -s)" in
    Linux*) OS=linux ;;
    Darwin*) OS=macos ;;
    *) OS=windows ;;
esac

SCRIPT_DIR="$(realpath "$(dirname "$BASH_SOURCE")")"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"
BIN_DIR="$(realpath "$1")"
MRBIND_DIR="$ROOT_DIR/thirdparty/mrbind"
OUT_DIR="$ROOT_DIR/build/python_over_c"
PYTHON="${PYTHON:-python3}"

# Paths for the native Windows tools, as is elsewhere.
native() { if [[ $OS == windows ]]; then cygpath -m "$1"; else echo "$1"; fi; }

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/pkg/meshlib"

# Like `MB_PB11_STRIPPED_NAMESPACES`, `MB_PB11_ADJUST_NAMES` and the pybind11-only ignores in the MeshLib headers.
GEN_FLAGS=(
    --replace-namespace MR ::
    --adjust-name '\bMR::' ''
    --hide-field id_
    --hide-field mask_
    # `mrcudapy` is a separate module.
    --ignore-c-file '^MRCCuda/'
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
# The pybind11 target names the types after the demangled `typeid()`, which spells the integer typedefs as their
#   underlying types; its build is parsed without the `size_t` support, so its docs and most of its aliases spell them too.
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
    # How the Itanium demangler spells these: the size of a nested `std::array` and the other `size_t` template arguments
    #   as `3ul`.
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

# `aliases.cpp` has the custom aliases of the pybind11 module; its `helpers.cpp` functions are in `helpers_over_c.py`.
ALIAS_FLAGS=()
while read -r alias target; do
    ALIAS_FLAGS+=(--alias "$alias" "$target")
done < <(sed -nE 's/^[[:space:]]*MR_ALIAS\([[:space:]]*([A-Za-z0-9_.]+)[[:space:]]*,[[:space:]]*([A-Za-z0-9_.]+)[[:space:]]*\);.*/\1 \2/p' "$SCRIPT_DIR/aliases.cpp" | tr -d '\r')

# The overrides are for local runs.
C2_DIR="${MESHLIBC2_DIR:-$ROOT_DIR/source/MeshLibC2}"
GEN_PYTHON="${MRBIND_GEN_PYTHON:-$MRBIND_DIR/build/mrbind_gen_python}"
if [[ $OS == windows ]]; then
    MSYS2_BIN="$(cygpath -u "${MSYS2_DIR:-C:/msys64}")/${MSYS2_ENV:-clang64}/bin"
    # Built in MSYS2, needs its runtime DLLs.
    PATH="$MSYS2_BIN:$PATH" "$GEN_PYTHON" \
        --input-json "$(native "$C2_DIR/temp/interop_desc.json")" --output-dir "$(native "$OUT_DIR/src")" \
        --clean-output-dir --module-name mrmeshpy --module-qualname meshlib.mrmeshpy "${GEN_FLAGS[@]}" "${ALIAS_FLAGS[@]}" \
        --extra-python "$(native "$SCRIPT_DIR/helpers_over_c.py")" --verbose 2> "$OUT_DIR/gen.log"
else
    "$GEN_PYTHON" \
        --input-json "$C2_DIR/temp/interop_desc.json" --output-dir "$OUT_DIR/src" \
        --clean-output-dir --module-name mrmeshpy --module-qualname meshlib.mrmeshpy "${GEN_FLAGS[@]}" "${ALIAS_FLAGS[@]}" \
        --extra-python "$SCRIPT_DIR/helpers_over_c.py" --verbose 2> "$OUT_DIR/gen.log"
fi
grep '^Generated' "$OUT_DIR/gen.log" || true

# The limited API, so one module serves every Python version.
PY_INCLUDE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
case $OS in
    linux|macos)
        if [[ -n "${LLVM_PREFIX:-}" ]]; then
            CXX="$LLVM_PREFIX/bin/clang++"
        else
            CXX="clang++-$(cat "$SCRIPT_DIR/clang_version.txt" | xargs)"
        fi
        MODULE_FILE="$OUT_DIR/pkg/meshlib/mrmeshpy.so"
        # E.g. `--gcc-install-dir`, like `EXTRA_CFLAGS` of `generate.mk`.
        read -r -a EXTRA_FLAGS <<< "${OVER_C_EXTRA_FLAGS:-}"
        C2_FILE="$BIN_DIR/libMeshLibC2.so"
        LINK_FLAGS=(-Wl,-rpath,"$BIN_DIR")
        if [[ $OS == macos ]]; then
            C2_FILE="$BIN_DIR/libMeshLibC2.dylib"
            # Like the pybind11 modules: the Python symbols come from the interpreter. Homebrew's Clang needs the SDK.
            LINK_FLAGS+=(-undefined dynamic_lookup -isysroot "$(xcrun --show-sdk-path)")
        fi
        time "$CXX" "$OUT_DIR/src/mrmeshpy.cpp" -o "$MODULE_FILE" \
            -shared -fPIC -std=c++20 -O1 -w -DPy_LIMITED_API=0x03080000 \
            -I"$OUT_DIR/src" -I"$C2_DIR/include" -I"$PY_INCLUDE" \
            -L"$BIN_DIR" -lMeshLibC2 "${LINK_FLAGS[@]}" ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
        ;;
    windows)
        CLANG_CL="${CLANG_CL:-$MSYS2_BIN/clang-cl.exe}"
        MODULE_FILE="$OUT_DIR/pkg/meshlib/mrmeshpy.pyd"
        C2_FILE="$BIN_DIR/MeshLibC2.dll"
        C2_LIB="${MESHLIBC2_LIB:-$(ls "$BIN_DIR"/MeshLibC2.lib "$ROOT_DIR"/source/TempOutput/{bin,lib}/MeshLibC2.lib 2>/dev/null | head -1 || true)}"
        PY_LIBS="$("$PYTHON" -c 'import os, sys; print(os.path.join(sys.base_prefix, "libs"))')"
        # `python3.lib` is the stable ABI's `python3.dll`. MSVC's own runtime and STL, like the C bindings.
        time env MSYS2_ARG_CONV_EXCL="*" "$CLANG_CL" /nologo /c /std:c++20 /EHsc /O1 /MD /bigobj -w \
            -DPy_LIMITED_API=0x03080000 -DNOMINMAX \
            -I"$(native "$OUT_DIR/src")" -I"$(native "$C2_DIR/include")" -I"$PY_INCLUDE" \
            "$(native "$OUT_DIR/src/mrmeshpy.cpp")" /Fo"$(native "$OUT_DIR/mrmeshpy.obj")"
        # `lld-link` directly, like the CMake build: the clang-cl driver also asks for `clang_rt.builtins.lib`, which MSYS2 doesn't have.
        env MSYS2_ARG_CONV_EXCL="*" "$(dirname "$CLANG_CL")/lld-link.exe" /nologo /DLL /OUT:"$(native "$MODULE_FILE")" /IMPLIB:"$(native "$OUT_DIR/mrmeshpy.lib")" \
            "$(native "$OUT_DIR/mrmeshpy.obj")" "$(native "$C2_LIB")" /LIBPATH:"$PY_LIBS" python3.lib
        ;;
esac

# The package, with the same pure Python `mrmeshnumpy` as the pybind11 one.
if [[ $OS == windows ]]; then
    # Like the `windows-only` part of the pybind11 package's `__init__.py`: `MeshLibC2.dll` and its dependencies.
    printf 'import os\nos.add_dll_directory(%s)\ndel os\n' "'$(native "$BIN_DIR")'" > "$OUT_DIR/pkg/meshlib/__init__.py"
else
    : > "$OUT_DIR/pkg/meshlib/__init__.py"
fi
cp "$SCRIPT_DIR"/../../source/mrmeshnumpy/mrmeshnumpy.{py,pyi} "$OUT_DIR/pkg/meshlib/"

# `mrviewerpy` over this `mrmeshpy`, if built: the `mrviewerpy_over_c` target (`MESHLIB_BUILD_PYTHON_OVER_C`).
MODULES=(mrmeshpy mrmeshnumpy)
# On Windows it's in the CMake build's own `bin`, which the caller passes as `OVER_C_VIEWER_DIR`.
VIEWER_DIR="${OVER_C_VIEWER_DIR:-$BIN_DIR}/python_over_c/meshlib"
if compgen -G "$VIEWER_DIR/mrviewerpy*" > /dev/null; then
    cp "$VIEWER_DIR"/mrviewerpy.{so,pyd} "$OUT_DIR/pkg/meshlib/" 2>/dev/null || true
    # MeshLib's pybind11 loads these from the module's directory.
    cp "$BIN_DIR"/meshlib/*pybind11nonlimitedapi* "$OUT_DIR/pkg/meshlib/" || true
    MODULES+=(mrviewerpy)
fi

# Sizes of the binaries as in a wheel: stripped (except on Windows, where the debug info is in `.pdb`), and deflate-compressed.
PB_FILE="$(ls "$BIN_DIR"/meshlib/mrmeshpy.{so,pyd} 2>/dev/null | head -1 || true)"
"$PYTHON" - "$OS" "$(native "$PB_FILE")" "$(native "$MODULE_FILE")" "$(native "$C2_FILE")" > "$OUT_DIR/sizes.txt" <<'EOF'
import os, shutil, subprocess, sys, tempfile, zlib
strip = {"linux": ["strip", "--strip-unneeded"], "macos": ["strip", "-x"]}.get(sys.argv[1])
for path in sys.argv[2:]:
    stripped = os.path.join(tempfile.mkdtemp(), "f")
    shutil.copy(path, stripped)
    if strip:
        subprocess.run(strip + [stripped], check=True)
    data = open(stripped, "rb").read()
    print(f"{path}: {os.path.getsize(path)} bytes, stripped {len(data)}, compressed {len(zlib.compress(data, 6))}")
EOF
cat "$OUT_DIR/sizes.txt"

DUMP="$(native "$MRBIND_DIR/scripts/python_api_dump.py")"
for MODULE in "${MODULES[@]}"; do
    "$PYTHON" "$DUMP" dump meshlib.$MODULE --path "$(native "$BIN_DIR")" --strip meshlib.$MODULE. -o "$(native "$OUT_DIR/api_${MODULE}_pybind11.json")" || { echo "::warning::Can't import the pybind11 $MODULE"; continue; }
    "$PYTHON" "$DUMP" dump meshlib.$MODULE --path "$(native "$OUT_DIR/pkg")" --strip meshlib.$MODULE. -o "$(native "$OUT_DIR/api_${MODULE}_over_c.json")" || { echo "::warning::Can't import the $MODULE over the C bindings"; continue; }
    "$PYTHON" "$DUMP" diff "$(native "$OUT_DIR/api_${MODULE}_pybind11.json")" "$(native "$OUT_DIR/api_${MODULE}_over_c.json")" --limit 1000000 > "$OUT_DIR/api_${MODULE}_diff.txt" || true
    head -1 "$OUT_DIR/api_${MODULE}_diff.txt"
    grep '^== ' "$OUT_DIR/api_${MODULE}_diff.txt" || true
done

# The Python tests, like `run_python_test_script.py` runs them; under a virtual display for the viewer tests, as in CI.
XVFB=()
if command -v xvfb-run > /dev/null; then XVFB=(xvfb-run -a); fi
(cd "$ROOT_DIR/test_python" && MeshLibPyModulesPath="$(native "$OUT_DIR/pkg")" ${XVFB[@]+"${XVFB[@]}"} "$PYTHON" -m pytest -q -p no:cacheprovider -rA --tb=line --basetemp="$(native "$OUT_DIR/pytest_temp")" > "$OUT_DIR/tests.txt" 2>&1) || true
tail -1 "$OUT_DIR/tests.txt"
