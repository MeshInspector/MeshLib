cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRCommonPlugins CXX)

file(GLOB_RECURSE SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h")

file(GLOB UI_JSON "*.ui.json")
file(GLOB ITEMS_JSON "*.items.json")
IF(MR_EMSCRIPTEN)
  file(COPY ${UI_JSON} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
  file(COPY ${ITEMS_JSON} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
  file(COPY "resource" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
ELSE() # MR_EMSCRIPTEN
  file(COPY ${UI_JSON} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  file(COPY ${ITEMS_JSON} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  file(COPY "resource" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
ENDIF() # MR_EMSCRIPTEN

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

IF(MR_EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    tbb
    MRMesh
    MRIOExtras
    MRViewer
    glad
    glfw
    imgui
    jsoncpp
    spdlog
  )
ELSEIF(WIN32 OR MESHLIB_USE_VCPKG)
  find_package(glad REQUIRED)
  find_package(OpenVDB REQUIRED)

  target_link_libraries(${PROJECT_NAME} PRIVATE
    TBB::tbb
    MRMesh
    MRIOExtras
    MRViewer
    glad::glad
    glfw
    imgui
    JsonCpp::JsonCpp
    OpenVDB::openvdb
    spdlog::spdlog
    fmt::fmt
  )
  IF(NOT WIN32)
    find_package(clip REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE clip::clip)
  ENDIF()
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE CLIP_ENABLE_IMAGE=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    tbb
    MRMesh
    MRIOExtras
    MRViewer
    clip
    glad
    glfw
    imgui
    JsonCpp::JsonCpp
    openvdb
    spdlog
    fmt
  )
ENDIF()

IF(MESHLIB_BUILD_VOXELS)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRVoxels
  )
ELSE()
  target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VOXELS)
ENDIF()

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
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resource DESTINATION "${MR_RESOURCES_DIR}")

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
