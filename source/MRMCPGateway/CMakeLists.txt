project(MRMCPGateway CXX)

add_executable(${PROJECT_NAME}
  MRMCPGateway.cpp
  MRMCPGatewayBackend.cpp
  MRMCPGatewayCache.cpp
  MRMCPGatewayMlTransport.cpp
  MRMCPGatewaySpawn.cpp
  MRMCPGatewayUtf8.cpp
)

# Same fastmcpp acquisition strategy as MRMcp/CMakeLists.txt — bundled subdirectory
# on Windows/vcpkg and macOS, system package on Debian/Ubuntu.
IF(MESHLIB_USE_VCPKG OR APPLE)
  IF(NOT TARGET fastmcpp_core)
    set(FASTMCPP_DEPS_ADD_SUBDIRECTORY ON)
    add_subdirectory(../fastmcpp fastmcpp)
  ENDIF()
  target_link_libraries(${PROJECT_NAME} PRIVATE fastmcpp_core)
  target_include_directories(${PROJECT_NAME} PRIVATE
    ${MESHLIB_THIRDPARTY_DIR}/fastmcpp/include
    ${MESHLIB_THIRDPARTY_DIR}/cpp-httplib
    ${MESHLIB_THIRDPARTY_DIR}/nlohmann-json/include
  )
ELSE()
  find_package(fastmcpp REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE fastmcpp::fastmcpp_core)
ENDIF()

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)
