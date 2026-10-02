vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/fastmcpp
    REF b6a4e107eadb6e925d285b61f30160e478ded2ae
    SHA512 188a7bfca6818434b66c6524f14aec4cb7c9516494a72abda63dae53ac20fa6ba0acac546da1bbf4460efb58a43e9c6a9442dc6ef639bda1a8fca6270d67fd2c
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DFASTMCPP_BUILD_TESTS=OFF
        -DFASTMCPP_BUILD_EXAMPLES=OFF
        -DFASTMCPP_FETCH_CURL=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME fastmcpp
    CONFIG_PATH lib/cmake/fastmcpp
)
