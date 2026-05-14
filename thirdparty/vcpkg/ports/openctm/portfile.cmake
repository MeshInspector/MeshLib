vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/OpenCTM
    REF d79b2caf14ae3fd390464603b50b56ea3347b847
    SHA512 3d1cd4be285b50abb06b24398b79a0ebdfab9e0473eec64f026bf93084b6ddcc82ee499e990f672f597e2dea909dfa03ba960f89da8ce75c8a110a6093301834
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME OpenCTM CONFIG_PATH lib/cmake/OpenCTM)
