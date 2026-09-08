vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/OpenCTM
    REF 32135dbc7aa1239fca8dff1849bfca9c21d0342e
    SHA512 174ea24303355e2c04963d83a7fa64c38c0144d9606903fdac3eb16e733ccc6e15ef6795f63aa41c13eaa4494ee26c1392a1ea99b6bffcacf1036b87d30a511f
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME OpenCTM CONFIG_PATH lib/cmake/OpenCTM)
