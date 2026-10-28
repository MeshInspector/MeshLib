vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/fastmcpp
    REF 3b98eee5a3bdd697558a13208f27fddb6214bc7c
    SHA512 dfd5517893fac327c4066d618c262afd15236551a3a99dc9d5d5928f2a8abf596041a1645d50bb885affa9f53efa15e63f1a569fdf44894f8e4489328d12e33a
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
