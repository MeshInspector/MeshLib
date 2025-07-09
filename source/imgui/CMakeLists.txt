cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(imgui LANGUAGES CXX)

# Don't emit warnings.
IF(MSVC)
  string(REPLACE "/W4" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
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

target_compile_definitions(${PROJECT_NAME} PUBLIC IMGUI_DISABLE_OBSOLETE_FUNCTIONS)
target_compile_definitions(${PROJECT_NAME} PUBLIC IMGUI_ENABLE_FREETYPE)
target_compile_definitions(${PROJECT_NAME} PUBLIC IMGUI_USER_CONFIG=\"imgui/MRCustomImGuiConfig.h\")

IF(MR_EMSCRIPTEN)
  add_compile_definitions(IMGUI_IMPL_OPENGL_ES3)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_FREETYPE=1")
ELSE()
  find_package(freetype)
  IF(NOT freetype_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(Freetype REQUIRED IMPORTED_TARGET freetype2)
    IF(CMAKE_VERSION VERSION_LESS 3.18)
      set_target_properties(PkgConfig::Freetype PROPERTIES IMPORTED_GLOBAL TRUE)
    ENDIF()
    add_library(Freetype::Freetype ALIAS PkgConfig::Freetype)
  ENDIF()

  find_package(glfw3 CONFIG REQUIRED)

  target_link_libraries(${PROJECT_NAME} PRIVATE
    glfw
    Freetype::Freetype
    ${CMAKE_DL_LIBS}
  )
ENDIF()

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
