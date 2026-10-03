vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/fastmcpp
    REF b1bcdc6ee554cc78520791c0a9846ada07230eea
    SHA512 0fea97191ff577bb0a5db2b742148dd846e73f5428eb2e42997829edf3a30a1ebeaec8bdcaf3f3cee9c1d66222a373ec9fe8d5f0a1eb97ced02bbb364eff5b6b
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
