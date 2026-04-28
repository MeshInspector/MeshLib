project(MRMCPGateway CXX)

add_executable(${PROJECT_NAME}
  MRMCPGateway.cpp
  MRMCPGatewayBackend.cpp
  MRMCPGatewayCache.cpp
  MRMCPGatewayMlTransport.cpp
  MRMCPGatewaySpawn.cpp
)

find_package(fastmcpp REQUIRED)
target_link_libraries(${PROJECT_NAME}
  PRIVATE
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

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)
