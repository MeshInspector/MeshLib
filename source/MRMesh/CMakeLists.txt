project(MRMesh CXX)

option(MRMESH_NO_TIFF "Disable TIFF support" OFF)
option(MRMESH_NO_GTEST "Disable building Google tests" OFF)

IF(DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
  set(MRMESH_NO_GTEST ON)
ENDIF()

IF(MR_EMSCRIPTEN)
  set(MRMESH_NO_TIFF ON)
ENDIF()

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h" "*.hpp")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h)

# Emscripten's Boost package doesn't contain CMake config files
IF(EMSCRIPTEN)
  IF(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
  ENDIF()
  find_package(Boost 1.73 REQUIRED)
ELSE()
  find_package(Boost 1.73 CONFIG COMPONENTS headers REQUIRED)
ENDIF()
find_package(Eigen3 REQUIRED)
find_package(fmt 8 REQUIRED)
find_package(JsonCpp REQUIRED)
find_package(spdlog 1.9 REQUIRED)
find_package(TBB 2020.1 COMPONENTS tbb REQUIRED)
find_package(tl-expected REQUIRED)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Boost::headers
    Eigen3::Eigen
    fmt::fmt
    JsonCpp::JsonCpp
    spdlog::spdlog
    TBB::tbb
    tl::expected
  PRIVATE
    ${CMAKE_DL_LIBS}
)

# libzip on Emscripten is unable to resolve its dependencies by itself
IF(EMSCRIPTEN)
  find_package(MbedTLS 1.0 REQUIRED)
ENDIF()
find_package(ZLIB REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)
# libzip prior to version 1.11.4 has incorrect CMake config that requires libzip tools to be installed
IF(MESHLIB_USE_VCPKG OR APPLE OR EMSCRIPTEN)
  # vcpkg, homebrew, and submodule versions are already updated
  find_package(libzip REQUIRED)
ELSE()
  # on Ubuntu LTS (22.04, 24.04) we have to use pkg-config instead
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(libzip REQUIRED IMPORTED_TARGET libzip)
  add_library(libzip::zip ALIAS PkgConfig::libzip)
ENDIF()
target_link_libraries(${PROJECT_NAME} PRIVATE libzip::zip)

# zlib-ng (native mode, zng_ prefix) powers MRZlib's zlibCompressStream;
# decompression and the compressZip/libzip pipeline stay on stock zlib.
#
# Windows: use the proper CMake config package that vcpkg's zlib-ng port
# ships. Multi-config Debug/Release matters here: find_library() picks one
# .lib and uses it for every configuration, so a Debug build links against
# the release zlib-ng2.dll and the applocal-copy step doesn't copy it next
# to the .exe (STATUS_DLL_NOT_FOUND at launch, seen on msvc-2019 Debug
# CMake with the x64-windows-meshlib-iterator-debug triplet). The
# find_package(CONFIG) target has IMPORTED_CONFIGURATIONS=DEBUG;RELEASE
# and drives applocal-copy correctly.
#
# Other platforms: find_library + find_path. On Ubuntu apt and Emscripten
# we build zlib-ng from thirdparty/zlib-ng and its generated config has
# the same non-relocatable-include-path bug we hit with libdeflate (breaks
# under the Docker multi-stage COPY-shuffle). Plain library/path lookup
# sidesteps it, and those platforms are single-config so the multi-config
# concern above doesn't apply.
IF(WIN32)
  # Can't use find_package(zlib-ng CONFIG): the vcpkg zlib-ng 2.1.5 port
  # that vcpkg 2024.10.21 (pinned for msvc-2019 legs) ships doesn't install
  # a CMake config file. Can't use a plain find_library() either: it loses
  # Debug/Release selectivity, linking Debug MRMesh.dll against release
  # zlib-ng2.dll and breaking applocal-copy (see the diagnostic output on
  # the iterator-debug triplet, STATUS_DLL_NOT_FOUND at MeshViewer.exe).
  #
  # Manually construct an imported target with per-configuration IMPLIB +
  # LOCATION paths, locating the files directly under vcpkg's triplet
  # install tree. Gives applocal-copy the metadata it needs to copy the
  # correct DLL next to the .exe per configuration.
  find_path(LIBZLIBNG_INCLUDE_DIR NAMES zlib-ng.h REQUIRED)
  find_library(LIBZLIBNG_LIBRARY_RELEASE
    NAMES zlib-ng
    PATHS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib"
    NO_DEFAULT_PATH REQUIRED
  )
  find_library(LIBZLIBNG_LIBRARY_DEBUG
    NAMES zlib-ngd zlib-ng
    PATHS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/lib"
    NO_DEFAULT_PATH REQUIRED
  )
  find_program(LIBZLIBNG_DLL_RELEASE
    NAMES zlib-ng2.dll zlib-ng.dll
    PATHS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
    NO_DEFAULT_PATH REQUIRED
  )
  find_program(LIBZLIBNG_DLL_DEBUG
    NAMES zlib-ngd2.dll zlib-ng2.dll zlib-ng.dll
    PATHS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin"
    NO_DEFAULT_PATH REQUIRED
  )
  add_library(zlib-ng::zlib SHARED IMPORTED)
  set_target_properties(zlib-ng::zlib PROPERTIES
    IMPORTED_CONFIGURATIONS "DEBUG;RELEASE"
    IMPORTED_IMPLIB_RELEASE  "${LIBZLIBNG_LIBRARY_RELEASE}"
    IMPORTED_LOCATION_RELEASE "${LIBZLIBNG_DLL_RELEASE}"
    IMPORTED_IMPLIB_DEBUG    "${LIBZLIBNG_LIBRARY_DEBUG}"
    IMPORTED_LOCATION_DEBUG  "${LIBZLIBNG_DLL_DEBUG}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBZLIBNG_INCLUDE_DIR}"
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE zlib-ng::zlib)
ELSE()
  find_library(LIBZLIBNG_LIBRARY
    NAMES z-ng zlib-ng
    REQUIRED
  )
  find_path(LIBZLIBNG_INCLUDE_DIR
    NAMES zlib-ng.h
    REQUIRED
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBZLIBNG_LIBRARY})
  target_include_directories(${PROJECT_NAME} PRIVATE ${LIBZLIBNG_INCLUDE_DIR})
ENDIF()

# TODO: CMake config
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${MESHLIB_THIRDPARTY_DIR}/parallel-hashmap>
)

IF(APPLE)
  # for Boost.Stacktrace
  target_compile_definitions(${PROJECT_NAME} PUBLIC _GNU_SOURCE)

  find_library(COCOA_LIBRARY Cocoa ONLY)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${COCOA_LIBRARY})
ENDIF()

IF(EMSCRIPTEN)
  target_compile_definitions(${PROJECT_NAME} PUBLIC
    EIGEN_STACK_ALLOCATION_LIMIT=0
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")
ENDIF()

IF(NOT MRMESH_NO_TIFF)
  find_package(TIFF REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE TIFF::TIFF)
ENDIF()

IF(NOT MRMESH_NO_GTEST)
  find_package(GTest REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE GTest::gtest)
ENDIF()

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()

install(
  FILES ${HEADERS} "config_cmake.h"
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}"
  ARCHIVE DESTINATION "${MR_MAIN_LIB_DIR}"
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)
install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${MR_CONFIG_DIR}
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${MR_CONFIG_DIR}
)
