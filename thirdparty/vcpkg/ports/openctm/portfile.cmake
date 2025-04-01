vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/OpenCTM
    REF 03bf805afeee672447149b5185c73a203a17c3fb
    SHA512 aa5f87c47ea3f6114c611bf7a21eb5af96d6b8dc907c69a68e519412cb67455e18a425654db970b4147086af27b54e15d8337ca10611d2d37a02788073b4bdbe
    HEAD_REF master
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME OpenCTM)
