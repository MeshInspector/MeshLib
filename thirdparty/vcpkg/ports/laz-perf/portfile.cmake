vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/laz-perf
    REF 8d0a813a3b125fbc21ef214cf79609a1636cc9e4
    SHA512 56df3dce943c4e6865dcfb18b32f184d260867069f6f4bffe8f2478c813c13d618fcea3b121e9632aa008e09fbc6899858caaadcd45fb8f8e2e5793f34f1e2b2
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DWITH_TESTS=OFF
        -DWITH_EMBIND=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME LAZPERF)
