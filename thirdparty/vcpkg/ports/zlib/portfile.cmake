# MeshLib overlay port: "zlib" that actually builds zlib-ng in compat mode.
#
# Why this exists:
#   zlib-ng is a drop-in replacement for zlib with faster implementations
#   (SIMD-accelerated CRC32, optimized deflate hot loops). Its "compat"
#   build mode produces the same headers (zlib.h, zconf.h), the same
#   library name (libz), and the same CMake target (ZLIB::ZLIB) as zlib,
#   so any consumer depending on `zlib` picks it up without changes.
#
# Output on disk matches what the upstream `zlib` port would install:
#   include/zlib.h, include/zconf.h
#   lib/libz.so (or z.lib on Windows)
#   lib/cmake/ZLIB/ZLIBConfig*.cmake
#   lib/pkgconfig/zlib.pc
#
# The zlib-ng upstream vcpkg port (ports/zlib-ng) supports ZLIB_COMPAT as
# an opt-in triplet variable; here we hard-enable it so we always deliver
# compat-flavored outputs.

set(ZLIBNG_VERSION "2.3.3")
set(ZLIBNG_SHA512 "e2057c764f1d5aaee738edee7e977182c5b097e3c95489dcd8de813f237d92a05daaa86d68d44b331d9fec5d1802586a8f6cfb658ba849874aaa14e72a8107f5")

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO zlib-ng/zlib-ng
    REF "${ZLIBNG_VERSION}"
    SHA512 "${ZLIBNG_SHA512}"
    HEAD_REF develop
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DZLIB_COMPAT=ON
        -DZLIB_ENABLE_TESTS=OFF
        -DZLIBNG_ENABLE_TESTS=OFF
        -DWITH_GTEST=OFF
        -DWITH_NEW_STRATEGIES=ON
    OPTIONS_RELEASE
        -DWITH_OPTIM=ON
)
vcpkg_cmake_install()
vcpkg_copy_pdbs()

# Rewrite absolute build-tree paths in lib/pkgconfig/zlib.pc into
# ${prefix}-relative ones, so downstream pkg-config consumers (e.g. curl's
# PkgConfig::curl target in our Linux-vcpkg build) don't end up with
# imported-target include directories pointing into vcpkg's packages/
# staging area. On Windows the call acquires pkgconf via vcpkg_acquire_msys
# — the specific pinned msys2 build has been rotated out of msys2's live
# mirrors but is pre-staged under thirdparty/vcpkg/downloads/ (see the
# msys2-mingw-w64-x86_64-pkgconf-*.pkg.tar.zst file there), so the fetch
# hits the local copy and succeeds.
vcpkg_fixup_pkgconfig()

# In compat mode, CMake config files live under lib/cmake/ZLIB/ and expose
# the `ZLIB::ZLIB` imported target — same layout consumers of the upstream
# `zlib` port already expect.
vcpkg_cmake_config_fixup(PACKAGE_NAME ZLIB CONFIG_PATH lib/cmake/ZLIB)

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/share"
    "${CURRENT_PACKAGES_DIR}/debug/include"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.md")
