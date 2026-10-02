vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/fastmcpp
    REF 7403ec08a03ea0e9ec4eb320f7258cdcb84a1965
    SHA512 fe4d05c5a6d872c53e05a9f841d03e9a58277ad546252c1f3a51dd6e549a1239e402761d61549df699306f52d93abfe64109ec10e89736d5f45dd6d08c9d1e69
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
