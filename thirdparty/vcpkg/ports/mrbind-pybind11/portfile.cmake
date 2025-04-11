vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/mrbind-pybind11
    REF adb6e4b0562ed7c723dfb7243033d100e9facc81
    SHA512 d82b4748a0c1a987cd52f173132b920a5f23d28fe71f57e5994fd441acbf6d1a0792a360c1ec8cc11577c8e4546222f8d7b6f84a920c529d54d42800e55147c0
    HEAD_REF non-limited-api
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DPYBIND11_INSTALL=ON
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME pybind11)
