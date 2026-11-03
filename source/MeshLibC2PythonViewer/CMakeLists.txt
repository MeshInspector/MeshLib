# The C bindings of `source/MRPythonViewer` (generated into `include` and `src`), with that C++ code itself.
# Only `mrviewerpy` uses them, so they are not installed for C users. See `MeshLibC2Cuda` for the flags.

project(MeshLibC2PythonViewer CXX)

IF(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4297 /wd4305 /wd4804")
  string(REPLACE "/EHsc" "/EHs" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context")
ENDIF()

file(GLOB_RECURSE SOURCES "src/*.cpp" "../MRPythonViewer/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "../MRPythonViewer/*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  MRViewer
  MeshLibC2
)

IF(MESHLIB_BUILD_VOXELS)
  target_link_libraries(${PROJECT_NAME} PRIVATE MRVoxels)
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VOXELS)
ENDIF()

IF(APPLE)
  # `PyOS_InputHook`, from the interpreter that loads `mrviewerpy`.
  target_link_options(${PROJECT_NAME} PRIVATE "LINKER:-U,_PyOS_InputHook")
ENDIF()

target_include_directories(${PROJECT_NAME} PRIVATE
  include
  src
  ..
  ../..
  ../MeshLibC2/include
  ../MeshLibC2/src
)
target_compile_definitions(${PROJECT_NAME} PRIVATE MR_COMPILING_ANY_BINDINGS MR_COMPILING_C_BINDINGS)

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}"
  ARCHIVE DESTINATION "${MR_MAIN_LIB_DIR}"
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)
