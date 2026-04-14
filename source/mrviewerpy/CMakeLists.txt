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
