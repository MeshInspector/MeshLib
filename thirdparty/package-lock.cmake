# Version pins for every third-party library built by thirdparty/stage1 and thirdparty/stage2.
#
# Each entry is the CPMAddPackage argument list for one dependency, minus NAME, which the
# caller supplies: the stages via meshlib_add_package(), thirdparty/fetch by appending
# DOWNLOAD_ONLY YES. Deliberately not CPMDeclarePackage -- a declaration *replaces* the
# caller's arguments rather than extending them, which silently drops DOWNLOAD_ONLY.
#
# GIT_TAG is a release tag only where the pin sits exactly on one; everywhere else it is the
# commit hash, because the pin tracks a mid-branch commit or a MeshInspector fork. Bumping a
# hash pin onto a tag is a version change, not a cleanup.
#
# Editing a file under patches/ in place does not invalidate CPM's source cache -- its key
# covers a patch's path, not its contents. Clear CPM_SOURCE_CACHE (thirdparty_sources/ by
# default) after doing so, or the previously patched tree is reused.
#
# scripts/check_third_party_licenses.py parses this file, so keep one argument per line.

set(MESHLIB_PACKAGE_boost-libs
  VERSION 1.83.0
  URL https://github.com/boostorg/boost/releases/download/boost-1.83.0/boost-1.83.0.tar.xz
  PATCHES
    ${CMAKE_CURRENT_LIST_DIR}/patches/boost-locale-no-thread.patch
    ${CMAKE_CURRENT_LIST_DIR}/patches/boost-locale-std-mutex.patch
    ${CMAKE_CURRENT_LIST_DIR}/patches/boost-honor-skip-install-rules.patch
)

set(MESHLIB_PACKAGE_c-blosc
  GIT_REPOSITORY https://github.com/Blosc/c-blosc.git
  GIT_TAG v1.21.6
)

set(MESHLIB_PACKAGE_clip
  GIT_REPOSITORY https://github.com/dacap/clip.git
  GIT_TAG 7f2e86ab9690f7df88440002083edd257f87bc58
  PATCHES
    ${CMAKE_CURRENT_LIST_DIR}/patches/clip-config-use-project-name.patch
)

set(MESHLIB_PACKAGE_cpp-httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib
  GIT_TAG b045ee7f6b434a85fd011e96e28c6d4abfb18788
)

set(MESHLIB_PACKAGE_cpr
  GIT_REPOSITORY https://github.com/whoshuu/cpr.git
  GIT_TAG 1.14.2
)

set(MESHLIB_PACKAGE_eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5.0.1
)

set(MESHLIB_PACKAGE_expected
  GIT_REPOSITORY https://github.com/Developer-Ecosystem-Engineering/expected.git
  GIT_TAG d6e1fcc766b725f196d7f58097d72f8cfab4d56a
)

set(MESHLIB_PACKAGE_fastmcpp
  GIT_REPOSITORY https://github.com/MeshInspector/fastmcpp
  GIT_TAG 9aa1a179f886f6d098f67693e18ee4778db924b7
)

set(MESHLIB_PACKAGE_fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 47a66c5eccc0cce71ad81b4a681a2032d86ca951
)

set(MESHLIB_PACKAGE_GDCM
  GIT_REPOSITORY https://github.com/malaterre/GDCM.git
  GIT_TAG v3.2.6
)

set(MESHLIB_PACKAGE_glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG e86f90457371c6233053bacf0d6f486a51ddcd67
)

set(MESHLIB_PACKAGE_googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG b796f7d44681514f58a683a3a71ff17c94edb0c1
)

set(MESHLIB_PACKAGE_jsoncpp
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
  GIT_TAG 42e892d96e47b1f6e29844cc705e148ec4856448
)

set(MESHLIB_PACKAGE_laz-perf
  GIT_REPOSITORY https://github.com/MeshInspector/laz-perf
  GIT_TAG 05ea01542e5c4417c05e7222f920e06276c79324
)

set(MESHLIB_PACKAGE_libE57Format
  GIT_REPOSITORY https://github.com/MeshInspector/libE57Format
  GIT_TAG 46eb0d02e00a6b0aeaee3e6655328b78c7e07b5b
)

set(MESHLIB_PACKAGE_libjpeg-turbo
  GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo.git
  GIT_TAG 7fa4b5b762c9a99b46b0b7838f5fd55071b92ea5
  PATCHES
    ${CMAKE_CURRENT_LIST_DIR}/patches/libjpeg-turbo-allow-add-subdirectory.patch
)

set(MESHLIB_PACKAGE_libzip
  GIT_REPOSITORY https://github.com/nih-at/libzip.git
  GIT_TAG v1.11.4
)

set(MESHLIB_PACKAGE_mbedtls
  GIT_REPOSITORY https://github.com/Mbed-TLS/mbedtls
  GIT_TAG v3.5.0
)

set(MESHLIB_PACKAGE_nlohmann-json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG 3946872265598aed5a7aea68cad4d9d1f168bd4b
)

set(MESHLIB_PACKAGE_onetbb
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
  GIT_TAG 12ceae12138af08845b3e8c369b24527346fe99e
)

set(MESHLIB_PACKAGE_OpenCTM
  GIT_REPOSITORY https://github.com/MeshInspector/OpenCTM.git
  GIT_TAG 2b444a4c70c432c1f4990fb32b0fcb11d0d81aad
)

set(MESHLIB_PACKAGE_openvdb
  GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openvdb
  GIT_TAG 87a508ca48a69f06d149be6cb0d8289fc1314f72
)

set(MESHLIB_PACKAGE_parallel-hashmap
  GIT_REPOSITORY https://github.com/greg7mdp/parallel-hashmap.git
  GIT_TAG v2.0.0
)

set(MESHLIB_PACKAGE_spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG 6fa36017cfd5731d617e1a934f0e5ea9c4445b13
)

set(MESHLIB_PACKAGE_tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf
  GIT_TAG v2.8.2
)

set(MESHLIB_PACKAGE_tinyxml2
  GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
  GIT_TAG 321ea883b7190d4e85cae5512a12e5eaa8f8731f
)

set(MESHLIB_PACKAGE_zlib-ng
  GIT_REPOSITORY https://github.com/zlib-ng/zlib-ng.git
  GIT_TAG 860e4cff7917d93f54f5d7f0bc1d0e8b1a3cb988
)
