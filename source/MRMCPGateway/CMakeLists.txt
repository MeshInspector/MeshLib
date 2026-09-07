project(MRMCPGateway CXX)

IF(MSVC)
  # Disabling following warnings:
  #   warning C5204: class has virtual functions, but its trivial destructor is not virtual; instances of objects derived from this class may not be destructed correctly
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd5204")
ENDIF()

add_executable(${PROJECT_NAME}
  MRMCPGateway.cpp
  MRMCPGatewayBackend.cpp
  MRMCPGatewayCache.cpp
  MRMCPGatewayMlTransport.cpp
  MRMCPGatewaySpawn.cpp
  MRMCPGatewayUtf8.cpp
)

# Output binary name (project / target name stays MRMCPGateway for build-system
# wiring; installers and clients invoke `McpGateway[.exe]`).
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME McpGateway)

find_package(httplib REQUIRED)
find_package(fastmcpp REQUIRED)
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    httplib::httplib
    fastmcpp::fastmcpp_core
)
# TODO: why does it affect only Linux+vcpkg?
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND MESHLIB_USE_VCPKG)
  # fix missing references to 'getaddrinfo_a' and 'gai_*' functions (required by cpp-httplib)
  target_link_libraries(${PROJECT_NAME} PRIVATE anl)
endif()
# inject work-around for Clang libc++ and older Xcode versions' compatibility
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_precompile_headers(${PROJECT_NAME}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:"fastmcpp/clang.hpp">
  )
endif()

IF(MR_PCH)
  if(MSVC)
    # MSVC requires to match all defines exactly with precompiled header
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DFMT_SHARED /DSPDLOG_COMPILED_LIB /DSPDLOG_FMT_EXTERNAL /DSPDLOG_SHARED_LIB")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /DTBB_USE_DEBUG")
  endif()
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)
