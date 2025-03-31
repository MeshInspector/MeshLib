vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO dacap/clip
    #    REF "v${VERSION}"
    REF a1a5fd11b420ad321b000eba4f736d53ef557e89
    SHA512 ef65ed7de918190f2190e1e1bef1d5c2056f34be9e080d749e6a3b9e9db889327b21a1a7e23d6de1f14aef9a94e63fc86416bdf8567678c90d1b26427b1a9aed
    HEAD_REF main
    PATCHES
      "add-cmake-installation-rules.patch"
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DCLIP_EXAMPLES=OFF
        -DCLIP_TESTS=OFF
        -DCLIP_X11_WITH_PNG=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME clip)
