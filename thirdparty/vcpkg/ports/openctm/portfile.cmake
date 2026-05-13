vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/OpenCTM
    REF 2a036819689a43cb69d21f4be45c5c2abd4d4c9f
    SHA512 48b31cc8235469af7e7d9f5d486373ac69a1dc0b4f8b662a6ea37a27cf18a61f87e81f8274344826c7c6bbb1e6d4221dd1d55bb5d11c4907dba4b430bbc736c3
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME OpenCTM CONFIG_PATH lib/cmake/OpenCTM)
