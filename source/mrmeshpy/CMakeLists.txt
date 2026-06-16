# mrbind-generated Python bindings, compiled with the CMake compiler (MSVC on Windows) instead of
# clang. The sources are produced beforehand by `scripts/mrbind/generate.mk only-generate` (clang
# parse) + `scripts/mrbind/assemble_pybind_msvc.py`; they don't exist in a fresh checkout, so this
# project must be configured AFTER that generation step (it is not part of the main configure).
#
# Mirrors source/mrmeshnumpy/CMakeLists.txt (the .pyd module setup) and source/MeshLibC2/CMakeLists.txt
# (compiling mrbind-generated sources: no PCH, since the bindings add defines the shared PCH lacks).

project(mrmeshpy CXX)

# Only the per-fragment wrappers, helpers and aliases wrappers under src/ are compiled; the big
# generated source lives in generated/ and is pulled in via #include from the wrappers.
file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "src/*.h")

IF(NOT SOURCES)
  message(FATAL_ERROR
    "No generated sources in '${CMAKE_CURRENT_SOURCE_DIR}/src'. Generate them first:\n"
    "  make -f scripts/mrbind/generate.mk MODE=none only-generate ...\n"
    "  python3 scripts/mrbind/assemble_pybind_msvc.py --temp <TEMP_OUTPUT_DIR> --out source/mrmeshpy --fragments N")
ENDIF()

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib"
)

IF(WIN32)
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")
  # Build directly in `bin`, then copy to `bin/meshlib`, to keep Vcpkg from polluting `bin/meshlib`
  # with dependency DLLs (same trick as mrmeshnumpy).
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/meshlib"
  )
ENDIF()

IF(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
ENDIF()

target_include_directories(${PROJECT_NAME} PRIVATE
  src
  generated
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty
  ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/mrbind/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/mrbind
)

IF(MSVC)
  # mrbind's macro_sequence_for.h requires the conformant preprocessor (clang's is conformant by
  # default, so this only matters when building the bindings with MSVC). The generated bindings are
  # not meant to be warning-clean under the project's /Wall /WX, so don't treat warnings as errors.
  target_compile_options(${PROJECT_NAME} PRIVATE /Zc:preprocessor /WX-)
ENDIF()

# The remaining bindings defines (MRBIND_HEADER, MB_PB11_*, MR_COMPILING_*, ...) come from the
# generated src/mrbind_config.h that every TU includes; Py_LIMITED_API and the patched-pybind config
# come transitively from MRPython (as for mrmeshnumpy).
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    MRPython
    MRMesh
    MRIOExtras
    MRSymbolMesh
    ${CMAKE_DL_LIBS}
)

# Don't emit undefined references: we don't link a specific Python version (Py_LIMITED_API + the
# patched-pybind shim provide the symbols at runtime). Same as mrmeshnumpy.
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

IF(MESHLIB_BUILD_VOXELS)
  target_link_libraries(${PROJECT_NAME} PRIVATE MRVoxels)
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VOXELS)
ENDIF()

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION "${MR_PY_LIB_DIR}"
)
