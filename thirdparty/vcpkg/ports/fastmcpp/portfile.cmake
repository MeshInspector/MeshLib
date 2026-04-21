vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/fastmcpp
    REF 59e717403bf23113fc655284d482fcb8844b3595
    SHA512 cd97030888d5038a4b5e4313da003be2866a167c54499f4c831abb10ce0b876b08df3b3db926506d3bdc6b0fdcd6dae1cb4a0df203d3a19f5ca8669af4f8c7b1
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

vcpkg_cmake_config_fixup(PACKAGE_NAME fastmcpp)
