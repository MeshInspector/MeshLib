vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/OpenCTM
    REF eae87a4d9ac229bdc7c32492c84bd7cb1329e5c4
    SHA512 0e40fb168ce652e93b8fb7828cb4c34678026028cb03344fa847141a0dac0744bd17f9448464e9d1c35495dde5fd1efcd8b34d554ca31e1a04f9574e4e455bad
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME OpenCTM CONFIG_PATH lib/cmake/OpenCTM)
