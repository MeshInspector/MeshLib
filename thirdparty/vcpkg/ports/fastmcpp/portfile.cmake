vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/fastmcpp
    REF 658d9b522f4fd4436eed93d5d3533ba37eb6edbf
    SHA512 0ffe508c0708b325f1c1b16c59a7ab959b5098003aea84a351f0952abc0811bf2b52a2aa904f59b96c309b2bca72dc4c995776936a5942f909d6881cfe1e5d4e
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
vcpkg_copy_pdbs()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME fastmcpp
    CONFIG_PATH lib/cmake/fastmcpp
)
