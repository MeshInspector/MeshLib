vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/fastmcpp
    REF 3bacd94474b78488a9a8413e233eebac3d95d4fb
    SHA512 b85f5ec857a04f180cca7ec1f3b7a1b34eabdaea9d2a6ec35eded61f773fa14a9236de07a54199ad99b57fc77bc973ce65cd6b066c35334bb5365a9c3b02c83b
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
