vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO dacap/clip
    REF "v${VERSION}"
    SHA512 49b4c7d7a18b0fce4a00b350de7fd70c7b42bee47f3d475dcf767c5f3351b853aa951df3e771d7d1c6c3ba5b25849ccd9ceb08331ffc0f6c45c92db4c90fa1be
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
