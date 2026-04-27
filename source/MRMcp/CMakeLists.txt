project(MRMcp CXX)

file(GLOB HEADERS "*.h" "*.ipp")
file(GLOB SOURCES "*.cpp")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

find_package(fastmcpp REQUIRED
  HINTS "${MESHLIB_THIRDPARTY_ROOT_DIR}"
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    MRMesh
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
