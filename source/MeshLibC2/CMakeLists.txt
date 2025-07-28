cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MeshLibC2 CXX)

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  MRIOExtras
  MRSymbolMesh
  MRVoxels
  JsonCpp::JsonCpp
)

# Boost.stacktrace uses something from libdl in its headers.
target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})

IF(APPLE)
  include_directories(${jsoncpp_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE
    jsoncpp
  )
ELSEIF(MR_EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    jsoncpp
  )
ELSEIF(WIN32 OR MESHLIB_USE_VCPKG)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    JsonCpp::JsonCpp
  )
ELSE()
  include_directories(${jsoncpp_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE
    jsoncpp
  )
ENDIF()


IF(MR_EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    openvdb
  )
ELSEIF(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenVDB
  )
ELSEIF(WIN32 OR MESHLIB_USE_VCPKG)
  find_package(OpenVDB REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenVDB::openvdb
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    openvdb
  )
ENDIF()


IF(WIN32 OR MESHLIB_USE_VCPKG)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    TBB::tbb
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    tbb
  )
ENDIF()


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
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${MR_CONFIG_DIR}
)

install(FILES ${HEADERS} DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}")
install(FILES ${UI_JSON} DESTINATION "${MR_RESOURCES_DIR}")
install(FILES ${ITEMS_JSON} DESTINATION "${MR_RESOURCES_DIR}")

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
