set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

# Static-link zlib-ng so its symbols compile into libMRMesh.so directly.
# Avoids DT_VERNEED entries on libMRMesh.so pointing at libz-ng.so.2's
# ZLIB_NG_2.0.0/2.1.0 version nodes, which auditwheel's manylinux_2_28
# policy database doesn't recognise.
if(PORT STREQUAL "zlib-ng")
  set(VCPKG_LIBRARY_LINKAGE static)
endif()

set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)

set(VCPKG_FIXUP_ELF_RPATH ON)
