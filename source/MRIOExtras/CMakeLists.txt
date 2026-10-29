project(MRIOExtras CXX)

option(MRIOEXTRAS_NO_CTM "Disable OpenCTM support" OFF)
option(MRIOEXTRAS_NO_DRACO "Disable Google Draco support (.drc and compressed glTF)" OFF)
option(MRIOEXTRAS_NO_E57 "Disable E57 support" OFF)
option(MRIOEXTRAS_NO_GLTF "Disable glTF support" OFF)
option(MRIOEXTRAS_NO_JPEG "Disable JPEG support" OFF)
option(MRIOEXTRAS_NO_LAS "Disable LAS/LAZ support" OFF)
option(MRIOEXTRAS_NO_PDF "Disable PDF support" OFF)
option(MRIOEXTRAS_NO_PNG "Disable PNG support" OFF)
option(MRIOEXTRAS_NO_STEP "Disable STEP support" OFF)
option(MRIOEXTRAS_NO_TIFF "Disable TIFF support" OFF)
option(MRIOEXTRAS_NO_XML "Disable XML support (3MF, SVG)" OFF)

option(MRIOEXTRAS_OPENCASCADE_USE_XDE "Use OpenCASCADE XDE module" ON)

IF(MR_EMSCRIPTEN)
  # Sync those with `scripts/mrbind/generate.mk`.
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

IF(NOT MRIOEXTRAS_NO_XML)
  target_link_libraries(${PROJECT_NAME} PRIVATE tinyxml2)
ENDIF()

IF(NOT MRIOEXTRAS_NO_CTM)
  find_package(OpenCTM CONFIG REQUIRED
    HINTS "${MESHLIB_THIRDPARTY_ROOT_DIR}"
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenCTM::OpenCTM)
ENDIF()

IF(NOT MRIOEXTRAS_NO_DRACO)
  find_package(draco CONFIG REQUIRED
    HINTS "${MESHLIB_THIRDPARTY_ROOT_DIR}"
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE draco::draco)
ENDIF()

IF(NOT MRIOEXTRAS_NO_E57)
  find_package(E57Format CONFIG REQUIRED
    HINTS "${MESHLIB_THIRDPARTY_ROOT_DIR}"
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE E57Format)
ENDIF()

IF(NOT MRIOEXTRAS_NO_JPEG)
  IF(APPLE AND MESHLIB_USE_HOMEBREW)
    target_include_directories(${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${HOMEBREW_PREFIX}/opt/jpeg-turbo/include>) # @todo
    target_link_libraries(${PROJECT_NAME} PRIVATE ${HOMEBREW_PREFIX}/opt/jpeg-turbo/lib/libturbojpeg.dylib)
  ELSE()
    target_link_libraries(${PROJECT_NAME} PRIVATE turbojpeg)
  ENDIF()
ENDIF()

IF(NOT MRIOEXTRAS_NO_LAS)
  find_package(LAZPERF CONFIG REQUIRED
    HINTS "${MESHLIB_THIRDPARTY_ROOT_DIR}"
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE LAZPERF::lazperf)
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

  set(OpenCASCADE_REQUIRED_LIBRARIES
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
    list(APPEND OpenCASCADE_REQUIRED_LIBRARIES
      TKXDESTEP
      TKLCAF
      TKXCAF
    )
  ENDIF()
  foreach(LIBRARY_NAME ${OpenCASCADE_LIBRARIES})
    IF(${LIBRARY_NAME} IN_LIST OpenCASCADE_REQUIRED_LIBRARIES)
      target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARY_NAME})
    ENDIF()
  endforeach()
ENDIF()

IF(NOT MRIOEXTRAS_NO_TIFF)
  find_package(TIFF REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE TIFF::TIFF)
ENDIF()

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
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
