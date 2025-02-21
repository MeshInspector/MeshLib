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
    spdlog
    pybind11nonlimitedapi_stubs
    ${CMAKE_DL_LIBS}
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    spdlog::spdlog
    pybind11nonlimitedapi_stubs
    ${CMAKE_DL_LIBS}
  )
ENDIF()

IF(APPLE)
  target_link_options(${PROJECT_NAME} PRIVATE -undefined dynamic_lookup)
ENDIF()

# Don't emit undefined references. We need this because we don't want to link to a specific Python version.
# The flag we're removing here is something we only set on Linux, in `CompilerOptions.cmake`.
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

target_include_directories(${PROJECT_NAME} PUBLIC ${Python_INCLUDE_DIRS})

# `dladdr` in our patched `PYBIND11_MODULE(...)` needs this.
IF(NOT WIN32)
  target_compile_definitions(${PROJECT_NAME} PUBLIC
    _GNU_SOURCE
  )
ENDIF()

IF(MSVC AND DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
  # Copy Python DLLs into the bin directory.
  # We always copy the Release ones (pybind11 wants to use Release DLLs even in Debug mode).
  # By default only the `python311[_d].dll` gets copied, but we also need `python3.dll` and the non-debug `python311.dll` in Debug mode as said above.
  file(COPY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/python${Python_VERSION_MAJOR}.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  file(COPY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  # Need a Release build of Zlib too. In Release mode it usually gets copied with something else, but in Debug we must copy manually.
  # Might as well copy unconditionally here.
  file(COPY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/zlib1.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
ENDIF()

IF(WIN32)
  # Copy Windows-specific `__init__.py`. Since we also need to rename the file, we can't use `file(COPY ...)`.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/../../test_python/helper/win_init.py" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/../../test_python/helper/win_init.py" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib/__init__.py"
    COMMAND_ERROR_IS_FATAL ANY
  )
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
