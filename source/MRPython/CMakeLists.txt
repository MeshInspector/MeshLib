project(MRPython CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    MRMesh
)

IF(MSVC AND DEFINED VCPKG_INSTALLED_DIR AND DEFINED VCPKG_TARGET_TRIPLET)
  # Copy Python DLLs into the bin directory.
  # We always copy the Release ones: `MREmbeddedPython` uses them even in Debug mode.
  # By default only the `python311[_d].dll` gets copied, but we also need `python3.dll` and the non-debug `python311.dll` in Debug mode as said above.
  file(COPY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/python${Python_VERSION_MAJOR}.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  file(COPY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  # Release Python links to the Release zlib alias. In Release builds applocal
  # finds it via the consumer chain, but Debug builds only ask for Debug zlib
  # (zd.dll / zlib-ngd2.dll), so the Release `z.dll` never lands next to the
  # Release `python3XY.dll` we just copied above. Copy it manually so
  # `python3XY.dll`'s load-time `z.dll` import resolves and `Py_Initialize`
  # doesn't fail silently inside MRTest.exe. 
  if (EXISTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/z.dll")
    file(COPY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/z.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif()
  # for older vcpkg 2024.10.21 used by msvc-2019, which ship a standalone zlib port
  if (EXISTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/zlib1.dll")
    file(COPY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/zlib1.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif()
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
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
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
