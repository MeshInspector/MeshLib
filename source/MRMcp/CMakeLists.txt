project(MRMcp CXX)

file(GLOB HEADERS "*.h" "*.ipp")
file(GLOB SOURCES "*.cpp")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    MRMesh
)

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()

# On Windows, and on Linux+Vcpkg there is no third-party build script, so add fastmcpp as a subdirectory.
IF(MESHLIB_USE_VCPKG OR APPLE)
  IF(APPLE)
    set(FASTMCPP_DEPS_ADD_SUBDIRECTORY ON)
  ENDIF()
  add_subdirectory(../fastmcpp fastmcpp)
  target_link_libraries(${PROJECT_NAME} PRIVATE fastmcpp_core)
  target_include_directories(${PROJECT_NAME} PRIVATE
    ${MESHLIB_THIRDPARTY_DIR}/fastmcpp/include
    ${MESHLIB_THIRDPARTY_DIR}/cpp-httplib
  )
  target_include_directories(${PROJECT_NAME} PUBLIC
    ${MESHLIB_THIRDPARTY_DIR}/nlohmann-json/include
  )
ELSE()
  find_package(fastmcpp REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC fastmcpp::fastmcpp_core)
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
