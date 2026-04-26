set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)

set(VCPKG_FIXUP_ELF_RPATH ON)

# Build zlib-ng (overlaid as `zlib`) in compat mode. The empty-shim
# `thirdparty/vcpkg/ports/zlib` overlay depends on `zlib-ng`; this flag
# tells upstream's stock zlib-ng port to install ABI-compatible
# libz / zlib.h / ZLIB::ZLIB outputs, transparently shadowing stock zlib.
set(ZLIB_COMPAT ON)
