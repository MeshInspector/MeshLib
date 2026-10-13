set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_DISABLE_COMPILER_TRACKING ON)
set(VCPKG_CXX_FLAGS_DEBUG "${VCPKG_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=0")
set(VCPKG_C_FLAGS_DEBUG "${VCPKG_C_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=0")

# MSVC 19.51 arm64 at /O2 /Gs0 (OpenSSL's own flags) emits `bl __chkstk` before the prologue saves lr in
# tls_parse_all_extensions, which then returns into itself: every TLS handshake crashes. A later /Gs
# overrides /Gs0 and avoids it.
if(PORT STREQUAL "openssl")
    set(VCPKG_C_FLAGS_RELEASE "${VCPKG_C_FLAGS_RELEASE} /Gs")
    set(VCPKG_CXX_FLAGS_RELEASE "${VCPKG_CXX_FLAGS_RELEASE} /Gs") # vcpkg requires both or neither
endif()
