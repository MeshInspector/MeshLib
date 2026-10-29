vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/libE57Format
    REF 1f14b9ab86bd8599f19f8d79d8568b6a2548ebe8
    SHA512 5a6e3013de6cba37cc1c6fe1bf56859cab370862f865df064b5dd5e6740c22a75acf5d2e8788ffe5f6140ae5e372afa407632ab7b4248b34562a1627275f8665
    HEAD_REF tinyxml2-v3.4.1
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" E57_BUILD_SHARED)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DE57_BUILD_TEST=OFF
        -DE57_BUILD_SHARED=${E57_BUILD_SHARED}
        -DE57_RELEASE_LTO=OFF
        -DE57_XML_PARSER=TinyXML2
        -DCMAKE_DISABLE_FIND_PACKAGE_Git=1
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME e57format CONFIG_PATH "lib/cmake/E57Format")

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.md")
