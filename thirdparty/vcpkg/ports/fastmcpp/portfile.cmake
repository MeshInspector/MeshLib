vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/fastmcpp
    REF 796319c06f9bfc296cbcc31540f0b18e3ca344fb
    SHA512 b430d33112545f11eb089fdc458e91445140db633a13e111cbc7a3009f0c3d0b41ebbd8854d973ce23ed7c96dd6c58d30372c6572bda8a9ae64031ce32646b56
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
