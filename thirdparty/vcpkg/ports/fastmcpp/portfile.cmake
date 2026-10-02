vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/fastmcpp
    REF 1c07011600cf6c3474483c59bf97fed2c3d61b9e
    SHA512 cc8762f2da426ba7a0c4935fa5c8922ea0ed8aa5f30821d1521998a55bd3f44e401eb004bd9def53edc8b0aa01660d53d36337d6efd3e3150073b96de2f6d5ff
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
