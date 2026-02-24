project(MRViewer CXX)

option(MRVIEWER_NO_GTK "Disable GTK support (affects file dialog support on Linux)" OFF)
option(MRVIEWER_NO_LOCALE "Disable i18n support" OFF)
option(MRVIEWER_NO_VOXELS "Disable voxels support" OFF)
option(MRVIEWER_NO_XDG_DESKTOP_PORTAL "Disable XDG Desktop Portal support (affects file dialog support on Linux)" OFF)
option(MRVIEWER_WITH_BUNDLED_CURL "Link with cURL for extra functions required for portable Linux builds" OFF)

IF(NOT MESHLIB_BUILD_VOXELS)
  set(MRVIEWER_NO_VOXELS ON)
ENDIF()

IF(WIN32 OR APPLE OR EMSCRIPTEN)
  set(MRVIEWER_NO_GTK ON)
  set(MRVIEWER_NO_XDG_DESKTOP_PORTAL ON)
ENDIF()

file(GLOB HEADERS "*.h" "*.ipp")
file(GLOB SOURCES "*.cpp")
set(OBJCPP_SOURCES "")
IF(APPLE)
  file(GLOB OBJCPP_SOURCES "*.mm")
ENDIF()

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${OBJCPP_SOURCES} ${HEADERS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config_cmake.h)

IF(APPLE)
  list(APPEND CMAKE_FIND_ROOT_PATH ${MESHLIB_THIRDPARTY_ROOT_DIR})
ENDIF()

find_package(glad REQUIRED)
IF(NOT EMSCRIPTEN)
  find_package(glfw3 CONFIG REQUIRED)
ENDIF()

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    MRMesh
    glfw
    imgui
  PRIVATE
    MRIOExtras
    MRSymbolMesh
    glad::glad
    ${CMAKE_DL_LIBS}
)

IF(NOT EMSCRIPTEN)
  find_package(cpr REQUIRED)

  # prefer config over module for vcpkg compatibility
  set(CMAKE_FIND_PACKAGE_PREFER_CONFIG_PREV ${CMAKE_FIND_PACKAGE_PREFER_CONFIG})
  set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
  find_package(hidapi REQUIRED)
  set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ${CMAKE_FIND_PACKAGE_PREFER_CONFIG_PREV})

  target_link_libraries(${PROJECT_NAME}
    PRIVATE
      cpr::cpr
      hidapi::hidapi
  )
ENDIF()

IF(NOT WIN32 AND NOT EMSCRIPTEN)
  find_package(clip CONFIG REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE clip::clip)
ENDIF()

IF(APPLE)
  set_source_files_properties(${OBJCPP_SOURCES} PROPERTIES
    COMPILE_FLAGS "-x objective-c++"
    SKIP_PRECOMPILE_HEADERS ON
  )

  find_library(APPKIT_LIBRARY AppKit REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${APPKIT_LIBRARY})
ENDIF()

IF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(PkgConfig REQUIRED)

  IF(NOT MRVIEWER_NO_GTK)
    pkg_check_modules(gtk REQUIRED IMPORTED_TARGET gtk+-3.0)
    target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::gtk)
  ENDIF()

  IF(NOT MRVIEWER_NO_XDG_DESKTOP_PORTAL)
    pkg_check_modules(dbus REQUIRED IMPORTED_TARGET dbus-1)
    target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::dbus)
  ENDIF()

  IF(MRVIEWER_WITH_BUNDLED_CURL)
    pkg_check_modules(curl REQUIRED IMPORTED_TARGET libcurl)
    target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::curl)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MRVIEWER_WITH_BUNDLED_CURL)
  ENDIF()
ENDIF()

IF(NOT MRVIEWER_NO_VOXELS)
  target_link_libraries(${PROJECT_NAME} PRIVATE MRVoxels)
ENDIF()

IF(NOT MRVIEWER_NO_LOCALE)
  find_package(Boost COMPONENTS locale REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE Boost::locale)
ENDIF()

IF(MR_PCH)
  target_precompile_headers(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()

file(GLOB JSONS "*.json")
file(GLOB AWESOME_FONTS "${MESHLIB_THIRDPARTY_DIR}/fontawesome-free/*.ttf")
file(GLOB IMGUI_FONTS "${MESHLIB_THIRDPARTY_DIR}/imgui/misc/fonts/*.ttf")
file(GLOB MAIN_FONTS "${MESHLIB_THIRDPARTY_DIR}/Noto_Sans/*.*")
IF(EMSCRIPTEN)
  file(
    COPY
      ${JSONS}
      ${MAIN_FONTS}
      ${AWESOME_FONTS}
      ${IMGUI_FONTS}
      "resource"
    DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets"
  )
  file(
    COPY "wasm_async_func_list.txt"
    DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  )
ELSE()
  file(
    COPY
      ${JSONS}
      ${MAIN_FONTS}
      ${AWESOME_FONTS}
      ${IMGUI_FONTS}
      "resource"
    DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  )
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

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)
install(
  FILES
    ${MAIN_FONTS}
    ${AWESOME_FONTS}
    ${IMGUI_FONTS}
  DESTINATION ${MR_FONTS_DIR}
)
install(
  FILES ${JSONS}
  DESTINATION ${MR_RESOURCES_DIR}
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resource
  DESTINATION ${MR_RESOURCES_DIR}
)
IF(EMSCRIPTEN)
  install(
    FILES "wasm_async_func_list.txt"
    DESTINATION "${MR_RESOURCES_DIR}/wasm"
  )
ENDIF()

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)
