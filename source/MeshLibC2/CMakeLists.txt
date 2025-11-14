# Note that in this file I'm intentionally not enabling PCH, since this project adds some extra defines, and them missing in the PCH causes issues.
# They are bindings-only, and can't be added to the PCH itself.

project(MeshLibC2 CXX)

IF(MSVC)
  # Disabling following warnings:
  #   warning C4244: '+=': conversion from 'T' to 'unsigned short', possible loss of data
  #   warning C4305: 'argument': truncation from 'int' to 'T'
  #   warning C4804: '+=': unsafe use of type 'bool' in operation
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4305 /wd4804")
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context")
ENDIF()

IF(MSVC)
  # Don't assume that extern C functions don't throw exceptions, because ours can throw, if no error handler is installed.
  # The default flag is `/EHsc`, while we want only `/EHs`.
  string(REPLACE "/EHsc" "/EHs" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ENDIF()

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

IF(MESHLIB_C_BINDINGS_UNITY_BUILD)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_MODE BATCH
  )
ENDIF()

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  MRIOExtras
  MRSymbolMesh
  MRVoxels
  # Boost.stacktrace uses something from libdl in its headers.
  ${CMAKE_DL_LIBS}
)

#IF(WIN32)
#  # Patch IMath (a dependency of OpenVDB) to not include `/EHsc`. See the beginning of this file for explanation.
#  get_target_property(TMP_PROP Imath::Imath INTERFACE_COMPILE_OPTIONS)
#  IF(TMP_PROP)
#    string(REPLACE "/EHsc" "" TMP_PROP "${TMP_PROP}")
#    set_target_properties(Imath::Imath PROPERTIES INTERFACE_COMPILE_OPTIONS "${TMP_PROP}")
#  ENDIF()
#ENDIF()

target_include_directories(${PROJECT_NAME} PRIVATE include src ../..)
target_compile_definitions(${PROJECT_NAME} PRIVATE MR_COMPILING_ANY_BINDINGS MR_COMPILING_C_BINDINGS)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}"
  ARCHIVE DESTINATION "${MR_MAIN_LIB_DIR}"
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)

install(
  # Here the trailing slash removes the `include` part from the output path.
  DIRECTORY include/
  DESTINATION "${MR_INCLUDE_DIR}"
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
