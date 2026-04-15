project(MRMcp CXX)

file(GLOB HEADERS "*.h" "*.ipp")
file(GLOB SOURCES "*.cpp")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    MRMesh
    fastmcpp::fastmcpp_core
)

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()

# Fastmcpp stuff.
option(MESHLIB_BUILD_MCP "Enable MCP server" ON)
IF(MESHLIB_BUILD_MCP AND NOT MR_EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE fastmcpp_core)

  target_include_directories(${PROJECT_NAME} PRIVATE
    ${MESHLIB_THIRDPARTY_DIR}/fastmcpp/include
    ${MESHLIB_THIRDPARTY_DIR}/cpp-httplib
    ${MESHLIB_THIRDPARTY_DIR}/nlohmann-json/include
  )
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MR_ENABLE_MCP_SERVER=0)
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
