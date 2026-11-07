# x86_64 target built on an arm64 host: the self-hosted cross runner, see
# macos/crossplatform-builds/README.md. Same settings as x64-osx-meshlib; the
# separate name keeps the vcpkg caches (S3, GitHub) of the two legs apart.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

set(VCPKG_CMAKE_SYSTEM_NAME Darwin)
set(VCPKG_OSX_ARCHITECTURES x86_64)

set(VCPKG_BUILD_TYPE release)

set(VCPKG_OSX_DEPLOYMENT_TARGET 12.0)
set(VCPKG_OSX_SYSROOT macosx)
