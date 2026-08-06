vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/OpenCTM
    REF 2b444a4c70c432c1f4990fb32b0fcb11d0d81aad
    SHA512 36c16d42063dc732c1f47b711ce59395ced425907707190fe80088c0cd30ee83f5348960c7710b8ebeaa2ad0d9e2d928e60f4c5fd652e17559e0f08c72b15fd3
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME OpenCTM CONFIG_PATH lib/cmake/OpenCTM)
