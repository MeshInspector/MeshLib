vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO dacap/clip
    #    REF "v${VERSION}"
    REF 7f2e86ab9690f7df88440002083edd257f87bc58
    SHA512 b8d9deeabcebb3f1d09df25fe30519660bdfdc52f3261816a82af631eb65b974748546503e3c9df4f0407ed19c91a566f46c624f7d12fd201203f0af55a15411
    HEAD_REF main
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
