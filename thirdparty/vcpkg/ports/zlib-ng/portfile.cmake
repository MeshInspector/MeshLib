vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO zlib-ng/zlib-ng
    REF "${VERSION}"
    SHA512 e2057c764f1d5aaee738edee7e977182c5b097e3c95489dcd8de813f237d92a05daaa86d68d44b331d9fec5d1802586a8f6cfb658ba849874aaa14e72a8107f5
    HEAD_REF develop
)

# MeshLib: strip zlib-ng's GNU symbol version script and the matching .symver
# pragmas in its C sources.
#
# Upstream's CMakeLists.txt defines -DHAVE_SYMVER (which turns on __asm__(
# ".symver foo, foo@@ZLIB_NG_2.0.0") pragmas in zbuild.h) and passes
# -Wl,--version-script=zlib-ng.map to the linker whenever the target is
# non-Apple, non-AIX UNIX. Both together tag every exported symbol in
# libz-ng.so with ZLIB_NG_2.0.0 / ZLIB_NG_2.1.0 version nodes, which end up
# in DT_VERNEED of anything linking against libz-ng.
#
# auditwheel's manylinux policy database has no entry for (libz-ng.so.2,
# ZLIB_NG_*), so the MeshLib NuGet wheel-repair step fails with "too-recent
# versioned symbols" even though no actual symbol is too recent. We don't
# exercise zlib-ng's ABI-versioning machinery (our consumers rebuild against
# whatever libz-ng we ship), so we neutralize both knobs by flipping the
# guarding condition to FALSE. Upstream's zlib-ng.map file is left on disk
# but never wired into the build.
vcpkg_replace_string(
    "${SOURCE_PATH}/CMakeLists.txt"
    "if(NOT APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL AIX)"
    "if(FALSE)  # MeshLib: symbol versioning disabled, see thirdparty/vcpkg/ports/zlib-ng/portfile.cmake"
)

# Set ZLIB_COMPAT in the triplet file to turn on
if(NOT DEFINED ZLIB_COMPAT)
    set(ZLIB_COMPAT OFF)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        "-DZLIB_FULL_VERSION=${ZLIB_FULL_VERSION}"
        -DZLIB_ENABLE_TESTS=OFF
        -DWITH_NEW_STRATEGIES=ON
        -DZLIB_COMPAT=${ZLIB_COMPAT}
    OPTIONS_RELEASE
        -DWITH_OPTIM=ON
)
vcpkg_cmake_install()
vcpkg_copy_pdbs()

# Condition in `WIN32`, from https://github.com/zlib-ng/zlib-ng/blob/2.1.5/CMakeLists.txt#L1081-L1100
# (dynamic) for `zlib` or (static `MSVC) for `zlibstatic` or default `z`
# i.e. (windows) and not (static mingw) https://learn.microsoft.com/en-us/vcpkg/maintainers/variables#vcpkg_target_is_system
if(VCPKG_TARGET_IS_WINDOWS AND (NOT (VCPKG_LIBRARY_LINKAGE STREQUAL static AND VCPKG_TARGET_IS_MINGW)))
    set(_port_suffix)
    if(ZLIB_COMPAT)
        set(_port_suffix "")
    else()
        set(_port_suffix "-ng")
    endif()

    set(_port_output_name)
    if(VCPKG_LIBRARY_LINKAGE STREQUAL "dynamic")
        set(_port_output_name "zlib${_port_suffix}")
    else()
        set(_port_output_name "zlibstatic${_port_suffix}")
    endif()

    # CMAKE_DEBUG_POSTFIX from https://github.com/zlib-ng/zlib-ng/blob/2.1.5/CMakeLists.txt#L494
    if(NOT DEFINED VCPKG_BUILD_TYPE OR VCPKG_BUILD_TYPE STREQUAL "release")
        vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/lib/pkgconfig/zlib${_port_suffix}.pc" " -lz${_port_suffix}" " -l${_port_output_name}")
    endif()
    if(NOT DEFINED VCPKG_BUILD_TYPE OR VCPKG_BUILD_TYPE STREQUAL "debug")
        vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig/zlib${_port_suffix}.pc" " -lz${_port_suffix}" " -l${_port_output_name}d")
    endif()
endif()

vcpkg_fixup_pkgconfig()

if(ZLIB_COMPAT)
    set(_cmake_dir "ZLIB")
else()
    set(_cmake_dir "zlib-ng")
endif()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/${_cmake_dir})

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share"
                    "${CURRENT_PACKAGES_DIR}/debug/include"
)
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.md")
