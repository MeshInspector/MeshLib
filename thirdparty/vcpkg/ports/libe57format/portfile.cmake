vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/libE57Format
    REF 3822c9075090c3ca6a2cb453c72ae17e33b2c46c
    SHA512 996b1abd3f7dc6bc0dedc1d9118d87300b639ab93829e97f55ba5aabffdd0ea2dcb550e3a4100536a5bc7f9ce4f545da3f306aa5e0d684cefd9630304fbc5e86
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
