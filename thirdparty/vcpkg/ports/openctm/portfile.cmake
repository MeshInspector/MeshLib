vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/OpenCTM
    REF 838a510daac28af2f2660f9215d580cbb341cee8
    SHA512 c1a19c797bb815d26096c14be200479b851328f82b595ab871299af10b93c68b93634e6ad87b2bdd2ee5a9b468ef955a2af27a158ab7a48bf32c70a3851f63aa
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME OpenCTM CONFIG_PATH lib/cmake/OpenCTM)
