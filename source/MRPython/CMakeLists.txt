cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRPython CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

# TODO: find Python independently

IF(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    Python::Python
    spdlog
    pybind11nonlimitedapi_stubs
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    Python::Python
    spdlog::spdlog
    pybind11nonlimitedapi_stubs
  )
ENDIF()

# It's a good idea to have this match `PYTHON_MIN_VERSION` in `scripts/mrbind/generate.mk`.
# Here `0x030800f0` corresponds to 3.8 (ignore the `f0` suffix at the end, it just means a release version as opposed to alpha/beta/etc).
target_compile_definitions(${PROJECT_NAME} PRIVATE Py_LIMITED_API=0x030800f0)

# It's a good idea to have this match the value specified in `scripts/mrbind/generate.mk`. See that file for the explanation.
target_compile_definitions(${PROJECT_NAME} PRIVATE PYBIND11_INTERNALS_VERSION=5)

IF(WIN32 AND MINGW)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MRPYTHON_EXPORT)
ENDIF()

IF(MR_PCH)
  # TODO: compare performance
  #target_precompile_headers(${PROJECT_NAME} PRIVATE
  #  "$<$<COMPILE_LANGUAGE:CXX>:MRPch/MRPch.h>"
  #  "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRPybind11.h>"
  #)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
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
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${MR_CONFIG_DIR}
)

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)
