project(MREmbeddedPython CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  MRPython
)

IF(WIN32 AND Python_LIBRARY_RELEASE)
  # The Release Python even in Debug builds (see the include of `Python.h`); its headers also link it by name.
  get_filename_component(PYTHON_RELEASE_LIB_DIR "${Python_LIBRARY_RELEASE}" DIRECTORY)
  target_link_directories(${PROJECT_NAME} PRIVATE "${PYTHON_RELEASE_LIB_DIR}")
  target_link_libraries(${PROJECT_NAME} PRIVATE "${Python_LIBRARY_RELEASE}")
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE Python::Python)
ENDIF()

target_include_directories(${PROJECT_NAME} PRIVATE ${Python_INCLUDE_DIRS})

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()

IF(WIN32)
  # The standard library's extension modules (_socket, _ssl, _sqlite3, ...) go to DLLs/, which Python puts on sys.path;
  # the DLLs they need go next to the executable, where most of them already are and the loader finds them
  IF(DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
    set(PYTHON_DLLS_DIR ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/python3/DLLs)
    file(GLOB PYTHON_EXTENSION_MODULES ${PYTHON_DLLS_DIR}/*.pyd)
    file(GLOB PYTHON_EXTENSION_DEPENDENCIES ${PYTHON_DLLS_DIR}/*.dll)
    file(COPY ${PYTHON_EXTENSION_MODULES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DLLs)
    file(COPY ${PYTHON_EXTENSION_DEPENDENCIES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    # The pure-Python standard library, python3XY.zip, built from the CPython sources of vcpkg's python3 port
    # by that port's own python.exe; rebuilt only when the script or that Python changes
    set(VCPKG_PYTHON ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/tools/python3/python.exe)
  ENDIF()
  IF(DEFINED VCPKG_PYTHON AND EXISTS ${VCPKG_PYTHON})
    set(PYTHON_ZIP_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/update_python_zip.py)
    set(PYTHON_ZIP ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.zip)
    add_custom_command(
      OUTPUT ${PYTHON_ZIP}
      COMMAND ${VCPKG_PYTHON} ${PYTHON_ZIP_SCRIPT} --output ${PYTHON_ZIP}
      DEPENDS ${PYTHON_ZIP_SCRIPT} ${VCPKG_PYTHON}
    )
    add_custom_target(MREmbeddedPythonZip ALL DEPENDS ${PYTHON_ZIP})
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
