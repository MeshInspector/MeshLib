# REF must track the `thirdparty/mrbind-pybind11` submodule commit: when bumping the submodule,
# update REF + SHA512 and bump "port-version" in vcpkg.json so the binary caches are invalidated.
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/mrbind-pybind11
    REF 2bead442a0146befdd5383bab79f5aa101afb573
    SHA512 379962547492cd8c1eee8dcd7dcc798911fadadec9656629e788f39faade19f59f82f42fc3ca4855298f9b0b254e315efdbe6627124a4b4bdea0fc1180c52db1
    HEAD_REF non-limited-api
)

set(EXTRA_OPTIONS "")
if(VCPKG_TARGET_IS_WINDOWS)
    # Pin the vcpkg Python; otherwise FindPython may pick a host installation.
    list(APPEND EXTRA_OPTIONS "-DPython_EXECUTABLE=${CURRENT_INSTALLED_DIR}/tools/python3/python.exe")
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DPYBIND11_INSTALL=ON
        -DPYBIND11_TEST=OFF
        -DPYBIND11_NONLIMITEDAPI_BUILD_STUBS=ON
        -DPYBIND11_NONLIMITEDAPI_INSTALL_EXPORTS=ON
        -DPYBIND11_NONLIMITEDAPI_SUFFIX=meshlib
        -DPYBIND11_NONLIMITEDAPI_PYTHON_MIN_VERSION_HEX=0x030800f0
        -DPYBIND11_NONLIMITEDAPI_INTERNALS_VERSION=5
        -DPYBIND11_NONLIMITEDAPI_COMPILER_TYPE_STRING=_meshlib
        -DPYBIND11_NONLIMITEDAPI_BUILD_ABI_STRING=_meshlib
        ${EXTRA_OPTIONS}
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
    PACKAGE_NAME pybind11nonlimitedapi
    CONFIG_PATH lib/cmake/pybind11nonlimitedapi
)
vcpkg_cmake_config_fixup(
    PACKAGE_NAME pybind11
    CONFIG_PATH share/cmake/pybind11
)
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include" "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
