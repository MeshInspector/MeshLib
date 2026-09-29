# the upstream vcpkg port without ${ALGORITHM_INCLUDE_PATCH} (merged in main) and with install-linkage.diff rebased onto main
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO google/draco
    REF 15bdb3a4f15a7a8d77489ac348a7a50d93de17a3
    SHA512 a896ab7b9f0af43c3389c857bf780ee983d213b8bd6137574348a0d2621b77222e8cf9088bd1047f8cca384c5d79ba9c3229c611288ee9b2bc77c51ba5816502
    HEAD_REF main
    PATCHES
        fix-compile-error-uwp.patch
        fix-uwperror.patch
        fix-pkgconfig.patch
        disable-symlinks.patch
        install-linkage.diff
)

if(VCPKG_TARGET_IS_EMSCRIPTEN)
    set(ENV{EMSCRIPTEN} "${EMSCRIPTEN_ROOT}")
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DPYTHON_EXECUTABLE=: # unused with DRACO_JS_GLUE off
        -DDRACO_JS_GLUE=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH share/cmake/draco)
vcpkg_fixup_pkgconfig()

# Install tools and plugins
if(NOT VCPKG_TARGET_IS_EMSCRIPTEN)
    vcpkg_copy_tools(TOOL_NAMES draco_encoder draco_decoder AUTO_CLEAN)
endif()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include" "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_copy_pdbs()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
