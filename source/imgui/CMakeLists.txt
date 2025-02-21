cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(imgui)

set(IMGUI_ROOT "../../thirdparty/imgui")

file(GLOB IMGUI_SOURCES
  "${IMGUI_ROOT}/*.cpp"
  "${IMGUI_ROOT}/misc/freetype/*.cpp"
  "${IMGUI_ROOT}/misc/cpp/*.cpp"
  "${IMGUI_ROOT}/backends/imgui_impl_opengl3.cpp"
  "${IMGUI_ROOT}/backends/imgui_impl_glfw.cpp"
  "*.cpp"
)

file(GLOB IMGUI_HEADERS
  "${IMGUI_ROOT}/*.h"
  "${IMGUI_ROOT}/misc/freetype/*.h"
  "${IMGUI_ROOT}/backends/imgui_impl_opengl3.h"
  "${IMGUI_ROOT}/backends/imgui_impl_glfw.h"
  "${IMGUI_ROOT}/backends/imgui_impl_opengl3_loader.h"
  "*.h"
)

add_library(imgui SHARED ${IMGUI_SOURCES} ${IMGUI_HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE "..")
target_include_directories(${PROJECT_NAME} PRIVATE "${IMGUI_ROOT}")
target_include_directories(${PROJECT_NAME} PRIVATE "../../thirdparty/parallel-hashmap")

# Don't emit warnings.
string(REPLACE "/W4" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

target_compile_definitions(${PROJECT_NAME} PUBLIC IMGUI_USER_CONFIG=\"imgui/MRCustomImGuiConfig.h\")
target_compile_definitions(${PROJECT_NAME} PUBLIC IMGUI_ENABLE_FREETYPE)
target_compile_definitions(${PROJECT_NAME} PRIVATE MRIMGUI_EXPORT)



IF(MR_EMSCRIPTEN)
  add_compile_definitions(IMGUI_IMPL_OPENGL_ES3)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_FREETYPE=1")
ELSE()
  set(BUILD_SHARED_LIBS ON)
  find_package(freetype REQUIRED)
  include_directories(${FREETYPE_INCLUDE_DIRS})
  IF(APPLE)
    find_package(glfw3 CONFIG REQUIRED)
    target_link_libraries(imgui PRIVATE ${FREETYPE_LIBRARIES} glfw)
    set(INSTALL_GTEST ON)
    add_subdirectory(../../thirdparty/googletest ./googletest)
  ELSEIF(WIN32)
    find_package(glfw3 CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw Freetype::Freetype)
  ELSE()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
  ENDIF()
ENDIF()

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    EXPORT ${PROJECT_NAME}
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/imgui/cmake
)
