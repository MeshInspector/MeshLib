# Overlay of vcpkg 2024.10.21 ports/boost-interprocess + an extra patch
# from boostorg/interprocess#224 ("Fix iterator invalidation bug",
# merged upstream 2024-08-07; not present in boost 1.86 shipped by
# vcpkg 2024.10.21). Used only by the VS2019 / 2024.10.21 Windows build
# (see thirdparty/install.bat and
# .github/actions/setup-vcpkg-windows/action.yml).

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO boostorg/interprocess
    REF boost-${VERSION}
    SHA512 f619d1e29e2ce2808d2edb6e022a877bcbf762ea1eb7bd2518dce2cd047be7402272f5e5bb7fc5120a7318f4643ab7107d908b4b7c15c4e3e0ee7231ed1fc7ee
    HEAD_REF master
    PATCHES
        unused-link-libs.diff
        fix-iterator-invalidation.diff
)

set(FEATURE_OPTIONS "")
boost_configure_and_install(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS ${FEATURE_OPTIONS}
)
