cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(imgui CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

file(GLOB IMGUI_SOURCES "${IMGUI_DIR}/*.cpp")
file(GLOB IMGUI_HEADERS "${IMGUI_DIR}/*.h")
set(IMGUI_BACKEND_FILES
  "${IMGUI_DIR}/backends/imgui_impl_glfw.h"
  "${IMGUI_DIR}/backends/imgui_impl_opengl3.h"
  "${IMGUI_DIR}/backends/imgui_impl_opengl3_loader.h"
  "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
  "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
)
set(IMGUI_MISC_FILES
  "${IMGUI_DIR}/misc/freetype/imgui_freetype.h"
  "${IMGUI_DIR}/misc/freetype/imgui_freetype.cpp"
  "${IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp"
)

add_library(${PROJECT_NAME} SHARED
  ${SOURCES}
  ${HEADERS}
  ${IMGUI_SOURCES}
  ${IMGUI_HEADERS}
  ${IMGUI_BACKEND_FILES}
  ${IMGUI_MISC_FILES}
)

IF(NOT MR_EMSCRIPTEN)
  find_package(Freetype REQUIRED)
  target_include_directories(${PROJECT_NAME} PRIVATE "${FREETYPE_INCLUDE_DIRS}")
ENDIF()

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    IMGUI_USER_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/MRCustomImGuiConfig.h"
    IMGUI_ENABLE_FREETYPE
)
IF(MR_EMSCRIPTEN)
  target_compile_definitions(${PROJECT_NAME}
    PRIVATE
      IMGUI_IMPL_OPENGL_ES3
  )
ENDIF()
IF(WIN32 AND MINGW)
  target_compile_definitions(${PROJECT_NAME}
    PRIVATE
      MRIMGUI_EXPORT
  )
ENDIF()

target_include_directories(${PROJECT_NAME} PRIVATE
  ${IMGUI_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  freetype
  glfw
  phmap::phmap
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${MR_MAIN_LIB_DIR}"
  ARCHIVE DESTINATION "${MR_MAIN_LIB_DIR}"
  RUNTIME DESTINATION "${MR_BIN_DIR}"
)

install(
  FILES ${HEADERS} ${IMGUI_HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)
