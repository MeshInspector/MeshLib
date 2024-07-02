#!/bin/bash

make -f _mrbind/scripts/apply_to_files.mk \
    INPUT_DIRS=source/MRMesh \
    INPUT_FILES_BLACKLIST='%/MRPython.h %/MREmbeddedPython.h %/MRIOFormatsRegistry.h %/MROpenVDBHelper.h %/MRRestoringStreamsSink.h %/MRDirectory.h %/MRVDBProgressInterrupter.h %/MRTupleBindings.h' \
    OUTPUT_DIR=build/binds \
    INPUT_GLOBS='*.h' \
    MRBIND='_mrbind/build/mrbind --ignore-pch-flags --ignore :: --allow MR --ignore MR::detail --ignore MR::Signal --ignore MR::UniquePtr --ignore MR::OpenVdbFloatGrid --ignore MR::RegisterRenderObjectConstructor --ignore MR::Config --allow std::integral_constant --skip-base boost::dynamic_bitset' \
    COMPILER_FLAGS="-std=c++20 -Wno-nonportable-include-path -Wno-enum-constexpr-conversion -Wno-deprecated-enum-enum-conversion -I. -Iinclude -Isource -I/usr/include/jsoncpp -isystem/usr/include/freetype2 -isystem/usr/include/gdcm-3.0 $(pkg-config --cflags python3-embed)" \
    COMPILER_FLAGS_LIBCLANG=-DMR_PARSING_FOR_PB11_BINDINGS \
    COMPILER="$CXX -I_mrbind/include -Wno-deprecated-declarations -Wno-implicitly-unsigned-literal -fPIC -DMR_COMPILING_PB11_BINDINGS -DMRBIND_HEADER='<mrbind/targets/pybind11.h>' -DMB_PB11_MODULE_NAME=mrmesh -DMB_PB11_ADJUST_NAMES='\"s/\\\\bMR:://;s/\\\\bvec_\\\\b/vec/\"' -DMB_PB11_ENABLE_CXX_STYLE_CONTAINER_METHODS" \
    LINKER_OUTPUT=build/Release/bin/mrmesh$(python3-config --extension-suffix) \
    LINKER="$CXX -fuse-ld=lld" \
    LINKER_FLAGS="$(pkg-config --libs python3-embed) -Lbuild/Release/bin -lMRMesh -Llib -ljsoncpp -lopenvdb -lTKernel -shared" \
    NUM_FRAGMENTS=4 \
    "$@"
