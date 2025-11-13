project(MRIOExtras CXX)

option(MRIOEXTRAS_NO_3MF "Disable 3MF support" OFF)
option(MRIOEXTRAS_NO_CTM "Disable OpenCTM support" OFF)
option(MRIOEXTRAS_NO_E57 "Disable E57 support" OFF)
option(MRIOEXTRAS_NO_GLTF "Disable glTF support" OFF)
option(MRIOEXTRAS_NO_JPEG "Disable JPEG support" OFF)
option(MRIOEXTRAS_NO_LAS "Disable LAS/LAZ support" OFF)
option(MRIOEXTRAS_NO_PDF "Disable PDF support" OFF)
option(MRIOEXTRAS_NO_PNG "Disable PNG support" OFF)
option(MRIOEXTRAS_NO_STEP "Disable STEP support" OFF)
option(MRIOEXTRAS_NO_TIFF "Disable TIFF support" OFF)
option(MRIOEXTRAS_NO_ZLIB "Disable Zlib usage" OFF)

option(MRIOEXTRAS_OPENCASCADE_USE_XDE "Use OpenCASCADE XDE module" ON)

IF(MR_EMSCRIPTEN)
  set(MRIOEXTRAS_NO_PDF ON)
  set(MRIOEXTRAS_NO_STEP ON)
  set(MRIOEXTRAS_NO_TIFF ON)
ENDIF()

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    MRMesh
)

IF(NOT MRIOEXTRAS_NO_3MF)
  target_link_libraries(${PROJECT_NAME} PRIVATE tinyxml2)
ENDIF()

IF(NOT MRIOEXTRAS_NO_CTM)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenCTM)
  IF(NOT MR_EMSCRIPTEN)
    target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${MESHLIB_THIRDPARTY_DIR}/OpenCTM-git/lib>)
  ENDIF()
ENDIF()

IF(NOT MRIOEXTRAS_NO_E57)
  target_link_libraries(${PROJECT_NAME} PRIVATE E57Format)

  IF(MSVC AND DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    file(GLOB XERCES_DLL_RELEASE "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/xerces-c_3_*.dll")
    file(GLOB XERCES_DLL_DEBUG "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/xerces-c_3_*.dll")

    # Not sure why this isn't auto-copied.
    IF($<CONFIG:Debug>)
      file(COPY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/E57Format-d.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
      IF(XERCES_DLL_DEBUG)
        message(STATUS "Copying Xerces Debug DLL: ${XERCES_DLL_DEBUG}")
        file(COPY ${XERCES_DLL_DEBUG} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
      ELSE()
        message(FATAL_ERROR "Xerces Debug DLL NOT FOUND in ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/! Build will fail.")
      ENDIF()
    ELSE()
      file(COPY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/E57Format.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
      IF(XERCES_DLL_RELEASE)
        message(STATUS "Copying Xerces Release DLL: ${XERCES_DLL_RELEASE}")
        file(COPY ${XERCES_DLL_RELEASE} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
      ELSE()
        message(FATAL_ERROR "Xerces Release DLL NOT FOUND in ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/! Build will fail.")
      ENDIF()
    ENDIF()
  ENDIF()
ENDIF()

IF(NOT MRIOEXTRAS_NO_JPEG)
  IF(APPLE)
    target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${HOMEBREW_PREFIX}/opt/jpeg-turbo/include>) # @todo
    target_link_libraries(${PROJECT_NAME} PRIVATE ${HOMEBREW_PREFIX}/opt/jpeg-turbo/lib/libturbojpeg.dylib)
  ELSE()
    target_link_libraries(${PROJECT_NAME} PRIVATE turbojpeg)
  ENDIF()
ENDIF()

IF(NOT MRIOEXTRAS_NO_LAS)
  target_link_libraries(${PROJECT_NAME} PRIVATE lazperf)
  IF(NOT MR_EMSCRIPTEN)
    target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${MESHLIB_THIRDPARTY_DIR}/laz-perf/cpp>)
  ENDIF()
ENDIF()

IF(NOT MRIOEXTRAS_NO_PDF)
  target_link_libraries(${PROJECT_NAME} PRIVATE hpdf)
ENDIF()

IF(NOT MRIOEXTRAS_NO_PNG)
  IF(MR_EMSCRIPTEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_LIBPNG=1")
  ELSE()
    find_package(PNG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE PNG::PNG)
  ENDIF()
ENDIF()

IF(NOT MRIOEXTRAS_NO_STEP)
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
  target_include_directories(${PROJECT_NAME} BEFORE PRIVATE ${OpenCASCADE_INCLUDE_DIR})
  target_link_directories(${PROJECT_NAME} BEFORE PRIVATE ${OpenCASCADE_LIBRARY_DIR})
  target_link_libraries(${PROJECT_NAME}
    PRIVATE
      TKernel
      TKBRep
      TKMath
      TKMesh
      TKDESTEP
      TKSTEP
      TKTopAlgo
      TKXSBase
  )
  IF(MRIOEXTRAS_OPENCASCADE_USE_XDE)
    target_link_libraries(${PROJECT_NAME}
      PRIVATE
        TKXDESTEP
        TKLCAF
        TKXCAF
    )
  ENDIF()
ENDIF()

IF(NOT MRIOEXTRAS_NO_TIFF)
  find_package(TIFF REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE TIFF::TIFF)
ENDIF()

IF(NOT MRIOEXTRAS_NO_ZLIB)
  find_package(ZLIB REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)
ENDIF()

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}"
  ARCHIVE DESTINATION "${MR_MAIN_LIB_DIR}"
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)

install(
  FILES ${HEADERS}
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
