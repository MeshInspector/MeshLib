#!/bin/bash

DIR="$(dirname -- "$0")"

make -f _mrbind/scripts/apply_to_files.mk \
    INPUT_DIRS=source/MRMesh \
    INPUT_FILES_BLACKLIST="$(cat "$DIR/input_file_blacklist.txt")" \
    OUTPUT_DIR=build/binds \
    INPUT_GLOBS='*.h' \
    MRBIND="_mrbind/build/mrbind $(cat "$DIR/mrbind_flags.txt")" \
    COMPILER_FLAGS="$(cat "$DIR/common_compiler_parser_flags.txt") $(pkg-config --cflags python3-embed)" \
    COMPILER_FLAGS_LIBCLANG="$(cat "$DIR/parser_only_flags.txt")" \
    COMPILER="$CXX $(cat "$DIR/compiler_only_flags.txt")" \
    LINKER_OUTPUT=build/Release/bin/mrmesh$(python3-config --extension-suffix) \
    LINKER="$CXX -fuse-ld=lld" \
    LINKER_FLAGS="$(pkg-config --libs python3-embed) -Lbuild/Release/bin -lMRMesh -shared \$(file <$DIR/linker_flags.txt)" \
    NUM_FRAGMENTS=4 \
    "$@"
