vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/mrbind-pybind11
    REF fdea944d10587b4ee1a4e743eecdf32fbb90b9fe
    SHA512 243c209f8007a2f030cc0108ad1b5236fbe7a00e07640a845821ec98c2db062cdb395e2f4d51c89808d272b16f43bae3daf666b5183301dffed54c6b8acd4f64
    HEAD_REF non-limited-api
    PATCHES
        fix-cmake-config.patch
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DPYBIND11_INSTALL=ON
        -DPYBIND11_NONLIMITEDAPI_SUFFIX=meshlib
        -DPYBIND11_NONLIMITEDAPI_BUILD_STUBS=ON
        -DPYBIND11_NONLIMITEDAPI_INSTALL_EXPORTS=ON
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME pybind11)
