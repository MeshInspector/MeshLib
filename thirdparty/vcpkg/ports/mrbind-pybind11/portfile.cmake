# REF must track the `thirdparty/mrbind-pybind11` submodule commit: when bumping the submodule,
# update REF + SHA512 and bump "port-version" in vcpkg.json so the binary caches are invalidated.
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO MeshInspector/mrbind-pybind11
    REF 9ca91973a7ce12a17cd73bd9328b052291906d8d
    SHA512 f59f99533132691311751656b1e7f36730313afec535c2d0026bd7e09ef12c2b978c70b644ad522540479828ae4a17060a751c92ec9fa852f8722a4111031bd5
    HEAD_REF non-limited-api
)

set(EXTRA_OPTIONS "")
if(VCPKG_TARGET_IS_WINDOWS)
    # Pin the vcpkg Python; otherwise FindPython may pick a host installation.
    list(APPEND EXTRA_OPTIONS "-DPython_EXECUTABLE=${CURRENT_INSTALLED_DIR}/tools/python3/python.exe")
else()
    # Unify the pybind ABI across GCC and Clang.
    list(APPEND EXTRA_OPTIONS
        -DPYBIND11_NONLIMITEDAPI_COMPILER_TYPE_STRING=_meshlib
        -DPYBIND11_NONLIMITEDAPI_BUILD_ABI_STRING=_meshlib
    )
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
