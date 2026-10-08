project(imgui LANGUAGES CXX)

# Don't emit warnings.
IF(MSVC)
  string(REPLACE "/Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
ELSE()
  string(REPLACE "-Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-Wextra" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-pedantic-errors" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 20)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nontrivial-memcall")
  ENDIF()
ENDIF()

set(IMGUI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/imgui")
IF(NOT EXISTS "${IMGUI_ROOT}/imgui.h")
  IF(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../.git")
    # initialize imgui submodule
    find_package(Git REQUIRED)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" submodule update --init --depth 1 -- thirdparty/imgui
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
      COMMAND_ERROR_IS_FATAL ANY
    )
  ELSE()
    # download imgui source code using CPM
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/cpm/cmake/CPM.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/cpm/package-lock.cmake)
    cmake_policy(PUSH)
    CPMAddPackage(NAME imgui ${MESHLIB_PACKAGE_imgui} DOWNLOAD_ONLY YES)
    cmake_policy(POP)
    set(IMGUI_ROOT "${imgui_SOURCE_DIR}")
  ENDIF()
ENDIF()
set(IMGUI_ROOT "${IMGUI_ROOT}" PARENT_SCOPE)

file(GLOB SOURCES
  "*.cpp"
  "${IMGUI_ROOT}/*.cpp"
  "${IMGUI_ROOT}/backends/imgui_impl_glfw.cpp"
  "${IMGUI_ROOT}/backends/imgui_impl_opengl3.cpp"
  "${IMGUI_ROOT}/misc/cpp/*.cpp"
  "${IMGUI_ROOT}/misc/freetype/*.cpp"
)

file(GLOB HEADERS
  "*.h"
  "${IMGUI_ROOT}/*.h"
)

file(GLOB BACKEND_HEADERS
  "${IMGUI_ROOT}/backends/imgui_impl_glfw.h"
  "${IMGUI_ROOT}/backends/imgui_impl_opengl3.h"
  "${IMGUI_ROOT}/backends/imgui_impl_opengl3_loader.h"
)

file(GLOB FREETYPE_HEADERS
  "${IMGUI_ROOT}/misc/freetype/*.h"
)

file(GLOB MISC_HEADERS
  "${IMGUI_ROOT}/misc/cpp/*.h"
)

add_library(imgui SHARED ${SOURCES} ${HEADERS} ${BACKEND_HEADERS} ${FREETYPE_HEADERS} ${MISC_HEADERS})

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${IMGUI_ROOT}>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
  IMGUI_DISABLE_OBSOLETE_FUNCTIONS
  IMGUI_ENABLE_FREETYPE
  IMGUI_USER_CONFIG=\"imgui/MRCustomImGuiConfig.h\"
)

IF(EMSCRIPTEN)
  target_compile_definitions(${PROJECT_NAME} PRIVATE
    IMGUI_IMPL_OPENGL_ES3
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_FREETYPE=1")
ELSE()
  find_package(glfw3 CONFIG REQUIRED)
ENDIF()

find_package(Freetype REQUIRED)

# ImGui doesn't depend on MRMesh directly but includes some of its files that require parallel-hashmap to be available
find_package(phmap 2.0 REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE phmap::phmap)

target_link_libraries(${PROJECT_NAME} PRIVATE
  glfw
  Freetype::Freetype
  ${CMAKE_DL_LIBS}
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}"
  ARCHIVE DESTINATION "${MR_MAIN_LIB_DIR}"
  INCLUDES DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)
install(
  FILES ${BACKEND_HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}/backends"
)
install(
  FILES ${FREETYPE_HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}/misc/freetype"
)
install(
  FILES ${MISC_HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}/misc/cpp"
)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${MR_CONFIG_DIR}
)

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION ${MR_CONFIG_DIR}
)
