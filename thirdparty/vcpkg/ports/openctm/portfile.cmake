vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/OpenCTM
    REF 49cfd0eb5de37cfd1d1290dc58d2b934dce9d387
    SHA512 ec1720655aa42edc744fcb77fd6981c2dc63a89acdbad3838b2fbd336d88052f7cd1cfcdacf4c3046d33f04a5008ae4ee981ff0e00aa9416000209a5ef3a5aea
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME OpenCTM CONFIG_PATH lib/cmake/OpenCTM)
