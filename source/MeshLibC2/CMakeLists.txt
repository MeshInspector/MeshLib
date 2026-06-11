# Note that in this file I'm intentionally not enabling PCH, since this project adds some extra defines, and them missing in the PCH causes issues.
# They are bindings-only, and can't be added to the PCH itself.


project(MeshLibC2 CXX)

IF(MSVC)
  # Disabling following warnings:
  #   warning C4242: '=': conversion from 'int' to 'unsigned short', possible loss of data
  #   warning C4244: '+=': conversion from 'T' to 'unsigned short', possible loss of data
  #   warning C4297: function assumed not to throw an exception but does
  #   warning C4305: 'argument': truncation from 'int' to 'T'
  #   warning C4355: 'this': used in base member initializer list
  #   warning C4574: '__has_feature' is defined to be '0': did you mean to use '#if __has_feature'?
  #   warning C4800: Implicit conversion from '_Ty *' to bool. Possible information loss
  #   warning C4804: '+=': unsafe use of type 'bool' in operation
  #   warning C5204: class has virtual functions, but its trivial destructor is not virtual; instances of objects derived from this class may not be destructed correctly
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4242 /wd4244 /wd4297 /wd4305 /wd4355 /wd4574 /wd4800 /wd4804 /wd5204")
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-in-bool-context -Wno-deprecated-declarations")
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

  # Spread the files evenly across the CPU cores, one batch per core.
  include(ProcessorCount)
  ProcessorCount(NUM_CORES)
  IF(NUM_CORES GREATER 0) # ProcessorCount() reports failure as 0; in that case keep CMake's default batch size.
    list(LENGTH SOURCES NUM_SOURCES)
    # Round up, to keep the batch count at most the core count, and to avoid a zero batch size
    # (which CMake interprets as "all files in a single batch").
    math(EXPR UNITY_BATCH_SIZE "(${NUM_SOURCES} + ${NUM_CORES} - 1) / ${NUM_CORES}")
    set_target_properties(${PROJECT_NAME} PROPERTIES UNITY_BUILD_BATCH_SIZE ${UNITY_BATCH_SIZE})
  ENDIF()
ENDIF()

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  MRIOExtras
  MRSymbolMesh
  MRVoxels
)

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
