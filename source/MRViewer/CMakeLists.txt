cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRViewer CXX)

file(GLOB HEADERS "*.h")
IF(APPLE)
  file(GLOB SOURCES "*.cpp" "*.mm")
ELSE()
  file(GLOB SOURCES "*.cpp")
ENDIF()

IF(MR_EMSCRIPTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")
ENDIF() # MR_EMSCRIPTEN

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

file(GLOB JSONS "*.json")
IF(MR_EMSCRIPTEN)
  file(COPY ${JSONS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
ELSE() # MR_EMSCRIPTEN
  file(COPY ${JSONS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
ENDIF() # MR_EMSCRIPTEN

IF(MR_EMSCRIPTEN)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    tbb
    jsoncpp
    imgui
    glfw
  )
ELSE() # NOT MR_EMSCRIPTEN
  IF(APPLE)
    # TODO: CMAKE_PREFIX_PATH for thirdparty
    #find_package(nfd REQUIRED)
    find_library(APPKIT_LIBRARY AppKit REQUIRED)

    add_definitions("-x objective-c++")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
      _GNU_SOURCE
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRMesh
      GTest::gtest
      fmt
      tbb
      OpenVDB
      spdlog
      jsoncpp
      cpr
      imgui
      glad
      glfw
      #nfd::nfd
      nfd
      hidapi
      ${APPKIT_LIBRARY}
    )
  ELSE() # NOT APPLE and NOT EMPSCRIPTEN
    find_package(GDCM CONFIG REQUIRED)
    pkg_check_modules(GTKMM REQUIRED gtkmm-3.0)

    target_link_libraries(${PROJECT_NAME} PRIVATE
      MRMesh
      gtest
      dl
      cpr
      imgui
      glad
      glfw
      gtkmm-3.0
      glib-2.0
      glibmm-2.4
      sigc-2.0
      hidapi-hidraw
      jsoncpp
      spdlog
      fmt
      openvdb
      tbb
      ${MR_PCH_PYTHON_LIB}
    )
    target_include_directories(${PROJECT_NAME} PRIVATE
      ${GTKMM_INCLUDE_DIRS}
    )
  ENDIF() # APPLE
ENDIF() # MR_EMSCRIPTEN

file(GLOB JSONS "*.json")
file(GLOB AWESOME_FONTS "${MESHLIB_THIRDPARTY_DIR}/fontawesome-free/*.ttf")
file(GLOB IMGUI_FONTS "${MESHLIB_THIRDPARTY_DIR}/imgui/misc/fonts/*.ttf")
IF(MR_EMSCRIPTEN)
  file(GLOB MAIN_FONTS "${MESHLIB_THIRDPARTY_DIR}/Noto_Sans/*.ttf")
  file(COPY ${MAIN_FONTS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
  file(COPY ${AWESOME_FONTS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
  file(COPY ${IMGUI_FONTS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
  file(COPY "resource" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
  file(COPY "wasm_async_func_list.txt" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
ELSE() # MR_EMSCRIPTEN
  file(GLOB MAIN_FONTS "${MESHLIB_THIRDPARTY_DIR}/Noto_Sans/*.*")
  file(COPY ${MAIN_FONTS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  file(COPY ${AWESOME_FONTS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  file(COPY ${IMGUI_FONTS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  file(COPY "resource" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
ENDIF() # MR_EMSCRIPTEN

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
install(FILES ${MAIN_FONTS} DESTINATION "${MR_FONTS_DIR}")
install(FILES ${AWESOME_FONTS} DESTINATION "${MR_FONTS_DIR}")
install(FILES ${IMGUI_FONTS} DESTINATION "${MR_FONTS_DIR}")
install(FILES ${JSONS} DESTINATION "${MR_RESOURCES_DIR}")
install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resource DESTINATION "${MR_RESOURCES_DIR}")

install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE MeshLib::
  DESTINATION ${MR_CONFIG_DIR}
)

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
