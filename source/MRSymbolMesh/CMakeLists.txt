cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRSymbolMesh CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

IF(MR_EMSCRIPTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_FREETYPE=1")
ELSE()
  find_package(Freetype REQUIRED)

  find_package(fmt REQUIRED)
  find_package(spdlog REQUIRED)

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(jsoncpp REQUIRED QUIET IMPORTED_TARGET jsoncpp)
ENDIF()

IF(MR_EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    freetype
    spdlog
    jsoncpp
  )
ELSEIF(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    Freetype::Freetype
    PkgConfig::jsoncpp
    fmt
    spdlog
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    Freetype::Freetype
    fmt::fmt
    spdlog::spdlog
    PkgConfig::jsoncpp
  )
ENDIF()

IF(WIN32 AND MINGW)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MRSYMBOLMESH_EXPORT)
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

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
