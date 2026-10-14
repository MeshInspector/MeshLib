project(MREmbeddedPython CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

# TODO: find Python independently

find_package(pybind11nonlimitedapi REQUIRED HINTS "${MESHLIB_THIRDPARTY_ROOT_DIR}/lib/cmake/pybind11nonlimitedapi")

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  MRPython
  pybind11nonlimitedapi::pybind11nonlimitedapi_stubs
  ${CMAKE_DL_LIBS}
)

IF(NOT WIN32)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python::Python)

  # `dladdr` in our patched `PYBIND11_MODULE(...)` needs this.
  target_compile_definitions(${PROJECT_NAME} PUBLIC _GNU_SOURCE)
ENDIF()

# Don't emit undefined references. We need this because we don't want to link to a specific Python version.
# The flag we're removing here is something we only set on Linux, in `CompilerOptions.cmake`.
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

if(NOT MESHLIB_USE_VCPKG)
  target_include_directories(${PROJECT_NAME} PUBLIC ${Python_INCLUDE_DIRS})
endif()

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()

# On Windows, copy the zipped Python environment to the output directory.
IF(WIN32)
  file(
    COPY ${MESHLIB_THIRDPARTY_DIR}/python/python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.zip
    DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  )
  # The standard library's extension modules (_socket, _ssl, _sqlite3, ...) go to DLLs/, which Python puts on sys.path;
  # the DLLs they need go next to the executable, where most of them already are and the loader finds them
  IF(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    set(PYTHON_DLLS_DIR ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/python3/DLLs)
    file(GLOB PYTHON_EXTENSION_MODULES ${PYTHON_DLLS_DIR}/*.pyd)
    file(GLOB PYTHON_EXTENSION_DEPENDENCIES ${PYTHON_DLLS_DIR}/*.dll)
    file(COPY ${PYTHON_EXTENSION_MODULES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DLLs)
    file(COPY ${PYTHON_EXTENSION_DEPENDENCIES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
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
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${MR_CONFIG_DIR}
)

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)
