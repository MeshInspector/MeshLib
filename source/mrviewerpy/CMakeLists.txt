project(mrviewerpy CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib"
)

IF(WIN32)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    SUFFIX ".pyd"
  )

  # We build directly in `bin`, then copy to `bin/meshlib`, to prevent Vcpkg from polluting `bin/meshlib` with the dependency DLLs.
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib"
  )
ENDIF()

IF(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  target_link_options(${PROJECT_NAME} PRIVATE -undefined dynamic_lookup)
ENDIF()

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    MRPython
    MRViewer
    ${CMAKE_DL_LIBS}
)

# Don't emit undefined references. We need this because we don't want to link to a specific Python version.
# The flag we're removing here is something we only set on Linux, in `CompilerOptions.cmake`.
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

IF(MESHLIB_BUILD_VOXELS)
  target_link_libraries(${PROJECT_NAME} PRIVATE MRVoxels)
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VOXELS)
ENDIF()

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION "${MR_PY_LIB_DIR}"
)

# Experimental: the same module over the `mrmeshpy` generated over the C bindings, see `scripts/mrbind/python_over_c.sh`.
IF(MESHLIB_BUILD_PYTHON_OVER_C)
  add_library(mrviewerpy_over_c SHARED ${SOURCES} ${HEADERS})
  set_target_properties(mrviewerpy_over_c PROPERTIES
    PREFIX ""
    OUTPUT_NAME mrviewerpy
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python_over_c/meshlib"
  )
  IF(WIN32)
    # A `.pyd` is a runtime output.
    set_target_properties(mrviewerpy_over_c PROPERTIES
      SUFFIX ".pyd"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/python_over_c/meshlib"
    )
  ENDIF()
  IF(APPLE)
    set_target_properties(mrviewerpy_over_c PROPERTIES SUFFIX ".so")
    target_link_options(mrviewerpy_over_c PRIVATE -undefined dynamic_lookup)
  ENDIF()
  target_include_directories(mrviewerpy_over_c PRIVATE "${MESHLIB_THIRDPARTY_DIR}/mrbind/include")
  target_compile_options(mrviewerpy_over_c PRIVATE
    $<IF:$<CXX_COMPILER_ID:MSVC>,/FI,-include> "${CMAKE_CURRENT_SOURCE_DIR}/MRPythonOverCTypes.h"
  )
  target_link_libraries(mrviewerpy_over_c PRIVATE MRPython MRViewer ${CMAKE_DL_LIBS})
  IF(MESHLIB_BUILD_VOXELS)
    target_link_libraries(mrviewerpy_over_c PRIVATE MRVoxels)
  ELSE()
    target_compile_definitions(mrviewerpy_over_c PRIVATE MESHLIB_NO_VOXELS)
  ENDIF()
ENDIF()
