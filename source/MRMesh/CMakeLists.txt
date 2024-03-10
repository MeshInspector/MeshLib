cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRMesh CXX)

option(MRMESH_NO_CLIPBOARD "Disable clipboard support for Linux and Mac" OFF)
option(MRMESH_NO_PDF "Disable PDF support" OFF)
option(MRMESH_NO_CPR "Disable CPR tests" OFF)
option(MRMESH_NO_PYTHON "Disable Python bindings" OFF)
option(MRMESH_NO_DICOM "Disable DICOM image support" OFF)
option(MRMESH_NO_OPENCTM "Disable OpenCTM support" OFF)
option(MRMESH_NO_LABEL "Disable support of label objects" OFF)
option(MRMESH_NO_JPEG "Disable JPEG support" OFF)
option(MRMESH_NO_PNG "Disable PNG support" OFF)
option(MRMESH_NO_TIFF "Disable TIFF support" OFF)
option(MRMESH_NO_VOXEL "Disable voxel support" OFF)
option(MRMESH_NO_GLTF "Disable glTF support" OFF)
option(MRMESH_NO_XML "Disable XML support (affects 3MF support)" OFF)
option(MRMESH_NO_E57 "Disable E57 support" OFF)
option(MRMESH_NO_LAS "Disable LAS/LAZ support" OFF)
option(MRMESH_NO_GTEST "Disable building Google tests" OFF)
option(MRMESH_NO_OPENCASCADE "Disable OpenCASCADE usage" OFF)
option(MRMESH_NO_ZLIB "Disable Zlib usage" OFF)

IF(DEFINED MESHLIB_PYTHON_SUPPORT AND NOT MESHLIB_PYTHON_SUPPORT)
  set(MRMESH_NO_PYTHON ON)
ENDIF()

IF(DEFINED BUILD_TESTING AND NOT BUILD_TESTING)
  set(MRMESH_NO_GTEST ON)
ENDIF()

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h)

set(MRMESH_OPTIONAL_DEPENDENCIES "")
set(MRMESH_OPTIONAL_LIBRARY_DIRS "")
set(MRMESH_OPTIONAL_INCLUDE_DIRS "")

IF(NOT MRMESH_NO_OPENCTM)
  list(APPEND MRMESH_OPTIONAL_DEPENDENCIES OpenCTM)
ENDIF()

IF(NOT MR_EMSCRIPTEN)
  IF(NOT MRMESH_NO_CLIPBOARD)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES clip)
  ENDIF()

  IF(NOT MRMESH_NO_PDF)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES hpdf)
  ENDIF()

  IF(NOT MRMESH_NO_CPR)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES cpr)
  ENDIF()

  IF(NOT MRMESH_NO_PYTHON)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES ${MR_PYTHON_LIB})
  ENDIF()

  IF(NOT MRMESH_NO_DICOM)
    find_package(GDCM CONFIG REQUIRED)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF)
  ENDIF()

  IF(NOT MRMESH_NO_LABEL)
    find_package(Freetype REQUIRED)
    include_directories(${FREETYPE_INCLUDE_DIRS})
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES freetype)
  ENDIF()

  IF(NOT MRMESH_NO_JPEG)
    IF(APPLE)
      target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${HOMEBREW_PREFIX}/opt/jpeg-turbo/include>) # @todo
      list(APPEND MRMESH_OPTIONAL_DEPENDENCIES ${HOMEBREW_PREFIX}/opt/jpeg-turbo/lib/libturbojpeg.dylib)
    ELSE()
      list(APPEND MRMESH_OPTIONAL_DEPENDENCIES turbojpeg)
    ENDIF()
  ENDIF()

  IF(NOT MRMESH_NO_PNG)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES png)
  ENDIF()

  IF(NOT MRMESH_NO_TIFF)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES tiff)
  ENDIF()

  IF(NOT MRMESH_NO_VOXEL)
    IF(APPLE)
      list(APPEND MRMESH_OPTIONAL_DEPENDENCIES OpenVDB)
    ELSE()
      list(APPEND MRMESH_OPTIONAL_DEPENDENCIES openvdb)
    ENDIF()
  ENDIF()

  IF(NOT MRMESH_NO_XML)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES tinyxml2)
  ENDIF()

  IF(NOT MRMESH_NO_E57)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES E57Format)
  ENDIF()

  IF(NOT MRMESH_NO_LAS)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES lazperf)
  ENDIF()

  IF(NOT MRMESH_NO_GTEST)
    IF(APPLE)
      list(APPEND MRMESH_OPTIONAL_DEPENDENCIES GTest::gtest)
    ELSE()
      list(APPEND MRMESH_OPTIONAL_DEPENDENCIES gtest)
    ENDIF()
  ENDIF()

  IF(NOT MRMESH_NO_OPENCASCADE)
    find_package(OpenCASCADE REQUIRED
      COMPONENTS
      # NOTE: the component order is significant
      FoundationClasses
      ModelingData
      ModelingAlgorithms
      Visualization
      ApplicationFramework
      DataExchange
    )
    set(OpenCASCADE_REQUIRED_LIBRARIES TKernel TKBRep TKMath TKMesh TKDESTEP TKSTEP TKXSBase)
    FOREACH(LIBRARY_NAME ${OpenCASCADE_LIBRARIES})
      IF(${LIBRARY_NAME} IN_LIST OpenCASCADE_REQUIRED_LIBRARIES)
        list(APPEND MRMESH_OPTIONAL_DEPENDENCIES ${LIBRARY_NAME})
      ENDIF()
    ENDFOREACH()
    list(APPEND MRMESH_OPTIONAL_LIBRARY_DIRS ${OpenCASCADE_LIBRARY_DIR})
    list(APPEND MRMESH_OPTIONAL_INCLUDE_DIRS ${OpenCASCADE_INCLUDE_DIR})
  ENDIF()

  IF(NOT MRMESH_NO_ZLIB)
    find_package(ZLIB REQUIRED)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES ZLIB::ZLIB)
  ENDIF()
ELSE()
  target_compile_definitions(${PROJECT_NAME}
    PRIVATE
    MRMESH_NO_PDF
    MRMESH_NO_CPR
    MRMESH_NO_PYTHON
    MRMESH_NO_DICOM
    MRMESH_NO_JPEG
    MRMESH_NO_VOXEL
    MRMESH_NO_E57
    MRMESH_NO_OPENCASCADE
  )

  IF(NOT MRMESH_NO_LAS)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES lazperf_s)
  ENDIF()

  IF(NOT MRMESH_NO_ZLIB)
    list(APPEND MRMESH_OPTIONAL_DEPENDENCIES z)
  ENDIF()
ENDIF()

# alternative Eigen include for external project
find_package(Eigen3 NO_MODULE)
IF(DEFINED EIGEN3_FOUND)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIRS}>)
  list(APPEND MRMESH_OPTIONAL_DEPENDENCIES Eigen3::Eigen)
ENDIF()

IF(NOT MR_EMSCRIPTEN)
  find_package(Boost COMPONENTS REQUIRED)
  find_package(TBB REQUIRED)
  pkg_check_modules(JSONCPP jsoncpp)
  include_directories(${JSONCPP_INCLUDE_DIRS})
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1 -s USE_FREETYPE=1 -s USE_LIBPNG=1")
ENDIF()

IF(APPLE OR MR_EMSCRIPTEN)
  #jpeg-turbo is keg-only, which means it was not symlinked into /opt/homebrew,
  #because libjpeg-turbo is not linked to prevent conflicts with the standard libjpeg.
  #include_directories(/opt/homebrew/opt/jpeg-turbo/include)
  #link_directories(/opt/homebrew/opt/jpeg-turbo/lib)

  IF(APPLE)
    find_library(COCOA_LIBRARY Cocoa ONLY)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _GNU_SOURCE)
    target_include_directories(${PROJECT_NAME} BEFORE PRIVATE
      ${MRMESH_OPTIONAL_INCLUDE_DIRS}
    )
    target_link_directories(${PROJECT_NAME} BEFORE PRIVATE
      ${MRMESH_OPTIONAL_LIBRARY_DIRS}
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE
      spdlog
      zip
      Boost::boost
      jsoncpp
      fmt
      tbb
      ${MRMESH_OPTIONAL_DEPENDENCIES}
      ${COCOA_LIBRARY}
    )
  ELSE()
    target_include_directories(${PROJECT_NAME} BEFORE PRIVATE
      ${MRMESH_OPTIONAL_INCLUDE_DIRS}
    )
    target_link_directories(${PROJECT_NAME} BEFORE PRIVATE
      ${MRMESH_OPTIONAL_LIBRARY_DIRS}
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE
      mbedtls
      mbedx509
      mbedcrypto
      zip
      freetype
      jsoncpp
      tbb
      ${MRMESH_OPTIONAL_DEPENDENCIES}
    )
  ENDIF()
ELSE()
  find_package(tl-expected REQUIRED)
  target_include_directories(${PROJECT_NAME} BEFORE PRIVATE
    ${MRMESH_OPTIONAL_INCLUDE_DIRS}
  )
  target_link_directories(${PROJECT_NAME} BEFORE PRIVATE
    ${MRMESH_OPTIONAL_LIBRARY_DIRS}
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE
    tl::expected
    fmt
    spdlog
    zip
    Boost::boost
    dl
    jsoncpp
    tbb
    ${MRMESH_OPTIONAL_DEPENDENCIES}
  )
ENDIF()

IF(WIN32 AND MINGW)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_EXPORT)
ENDIF()

set(CMAKE_INSTALL_PREFIX "/Library/Frameworks")
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}"
  ARCHIVE DESTINATION "${MR_MAIN_LIB_DIR}"
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)

install(
  FILES ${HEADERS} "config_cmake.h"
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${MR_CONFIG_DIR}
)

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)
