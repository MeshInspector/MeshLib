string(REPLACE "." "_" VERSION_STR "V${VERSION}")
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Open-Cascade-SAS/OCCT
    REF "${VERSION_STR}"
    SHA512 4ec271ec8db5f0d6f77ea5c0633b40334c796421806a344c568fb8d9ed942fec63f8dfcc65ab9f65e0446d5cd7a49beede5ac693421971b350e828ab1a19d773
    HEAD_REF master
    PATCHES
        fix-install-prefix-path.patch
        drop-bin-letter-d.patch
        dependencies.patch
        install-include-dir.patch
        remove-vcpkg-enabling.patch
        csf-redifinition.patch
)

if (VCPKG_LIBRARY_LINKAGE STREQUAL "dynamic")
    set(BUILD_TYPE "Shared")
else()
    set(BUILD_TYPE "Static")
endif()

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        eigen   USE_EIGEN
        tbb     USE_TBB
)

if ("lto" IN_LIST FEATURES)
    list(APPEND FEATURE_OPTIONS "-DBUILD_OPT_PROFILE=Production")
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DBUILD_LIBRARY_TYPE=${BUILD_TYPE}
        -DBUILD_CPP_STANDARD=C++20
        -DBUILD_RELEASE_DISABLE_EXCEPTIONS=ON
        -DBUILD_MODULE_ApplicationFramework=OFF
        -DBUILD_MODULE_DataExchange=OFF
        -DBUILD_MODULE_DETools=OFF
        -DBUILD_MODULE_Draw=OFF
        -DBUILD_MODULE_ModelingAlgorithms=OFF
        -DBUILD_MODULE_ModelingData=OFF
        -DBUILD_MODULE_Visualization=OFF
        -DBUILD_DOC_Overview=OFF
        -DBUILD_Inspector=OFF
        -DBUILD_ADDITIONAL_TOOLKITS="TKDESTEP;TKBinXCAF"
        -DINSTALL_DIR_LAYOUT=Unix
        -DINSTALL_DIR_DOC=share/trash
        -DINSTALL_DIR_SCRIPT=share/trash # not relocatable
        -DINSTALL_SAMPLES=OFF
        -DINSTALL_TEST_CASES=OFF
        -DUSE_DRACO=OFF
        -DUSE_FREETYPE=OFF
        -DUSE_FREEIMAGE=OFF
        -DUSE_OPENGL=OFF
        -DUSE_OPENVR=OFF
        -DUSE_GLES2=OFF
        -DUSE_RAPIDJSON=OFF
        -DUSE_TK=OFF
        -DUSE_VTK=OFF
        -DUSE_XLIB=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/opencascade)

#make occt includes relative to source_file
file(GLOB extra_headers
    LIST_DIRECTORIES false
    RELATIVE "${CURRENT_PACKAGES_DIR}/include/opencascade"
    "${CURRENT_PACKAGES_DIR}/include/opencascade/*.h"
)
list(JOIN extra_headers "|" extra_headers)
file(GLOB files "${CURRENT_PACKAGES_DIR}/include/opencascade/*.[hgl]xx")
foreach(file_name IN LISTS files)
    file(READ "${file_name}" filedata)
    string(REGEX REPLACE "(# *include) <([a-zA-Z0-9_]*[.][hgl]xx|${extra_headers})>" [[\1 "\2"]] filedata "${filedata}")
    file(WRITE "${file_name}" "${filedata}")
endforeach()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/share/opencascade/samples/qt")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/share/trash")

vcpkg_install_copyright(
    FILE_LIST
        "${SOURCE_PATH}/LICENSE_LGPL_21.txt"
        "${SOURCE_PATH}/OCCT_LGPL_EXCEPTION.txt"
)
