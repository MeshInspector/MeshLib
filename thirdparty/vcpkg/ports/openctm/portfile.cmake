vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/OpenCTM
    REF beb296897e42419fabbe19ace1386890f7794499
    SHA512 38d5b56ddf0c5f3c4409b02e97079959ccafacb611b5306789dbce0b0e3ff0ed02e997beec3fa68ed43108cc96d539bf7147618c41c4fbf4460f84170b459428
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME OpenCTM CONFIG_PATH lib/cmake/OpenCTM)
