project(MRVoxels CXX)

option(MRVOXELS_NO_DICOM "Disable DICOM format support" OFF)
option(MRVOXELS_NO_TIFF "Disable TIFF format support" OFF)

IF(MR_EMSCRIPTEN)
  set(MRVOXELS_NO_TIFF ON)
ENDIF()

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")
# MRPchCreate.cpp only drives the MSVC/MSBuild precompiled header; CMake generates its own.
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/MRPchCreate.cpp")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h)

find_package(OpenVDB 10 REQUIRED)

IF(MSVC)
  # Patch IMath (a dependency of OpenVDB) to not include `/EHsc` for C and CUDA compilers.
  # Upstream issue: https://github.com/AcademySoftwareFoundation/openexr/issues/2018
  get_target_property(IMATH_CXX_FLAGS Imath::Imath INTERFACE_COMPILE_OPTIONS)
  IF(IMATH_CXX_FLAGS)
    string(REPLACE "/EHsc" "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>" IMATH_CXX_FLAGS "${IMATH_CXX_FLAGS}")
    set_target_properties(Imath::Imath PROPERTIES INTERFACE_COMPILE_OPTIONS "${IMATH_CXX_FLAGS}")
  ENDIF()
ENDIF()

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    MRMesh
    OpenVDB::openvdb
)
# required for static linking
IF(EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE blosc)
ENDIF()

IF(NOT MRVOXELS_NO_DICOM)
  find_package(GDCM CONFIG REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF)
ENDIF()

IF(MR_PCH)
  # MSVC and Clang (>= 15) build an own PCH that also caches the heavy OpenVDB headers (the
  # dominant front-end cost in the voxels TUs). GCC keeps the lean shared PCH (a full PCH slows
  # GCC down: it loads the whole image per TU), as do AppleClang and older Clang.
  # MESHLIB_NO_VIEWER drops MRPch.h's viewer/imgui section, which MRVoxels (a lower layer)
  # neither needs nor has on its include path. See cmake/Modules/CompilerOptions.cmake.
  IF(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15))
    target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VIEWER)
    target_precompile_headers(${PROJECT_NAME} PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRVoxelsPch.h>")
  ELSE()
    target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
  ENDIF()
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

include(CMakePackageConfigHelpers)
configure_package_config_file(${PROJECT_NAME}Config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${MR_CONFIG_DIR}
  PATH_VARS MR_INCLUDE_DIR MR_MAIN_LIB_DIR
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${MR_CONFIG_DIR}
)

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)
