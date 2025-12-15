cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(example_plugin CXX)

if(EMSCRIPTEN)
    set(HTML_DIR "${CMAKE_BINARY_DIR}/html")
    # place built Wasm/JS files to `html' directory
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HTML_DIR})
endif()

find_package(MeshLib REQUIRED)
include_directories(${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
link_directories(${MESHLIB_THIRDPARTY_LIB_DIR})

add_library(MyPlugin MyPlugin.cpp)
target_link_libraries(MyPlugin
    PRIVATE
        MeshLib::MRViewer
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_executable(ViewerApp ViewerApp.cpp)
    if(NOT EMSCRIPTEN)
        target_link_libraries(ViewerApp
            PRIVATE
                MeshLib::MRViewer
        )
    else()
        # libraries and plugins must be linked with `--whole-archive' option for correct auto-initialization
        target_link_libraries(ViewerApp
            PRIVATE
                -Wl,--whole-archive
                MeshLib::MRViewer
                MeshLib::MRIOExtras
                MeshLib::MRCommonPlugins
                MyPlugin
                -Wl,--no-whole-archive
        )
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}")

        # place HTML/JS files to `html' directory
        file(GLOB MESHLIB_WASM_FILES "${MESHLIB_DATA_DIR}/wasm/*.*")
        file(GLOB LOCAL_WASM_FILES   "${CMAKE_CURRENT_SOURCE_DIR}/wasm/*.*")
        file(MAKE_DIRECTORY ${HTML_DIR})
        file(
            COPY
                ${MESHLIB_WASM_FILES}
                ${LOCAL_WASM_FILES}
            DESTINATION ${HTML_DIR}
        )

        # pack assets
        file(GLOB MESHLIB_JSONS "${MESHLIB_DATA_DIR}/*.json")
        file(GLOB MESHLIB_FONTS "${MESHLIB_DATA_DIR}/fonts/*.ttf")
        file(GLOB LOCAL_JSONS   "${CMAKE_CURRENT_SOURCE_DIR}/*.json")
        file(MAKE_DIRECTORY "${HTML_DIR}/assets")
        file(
            COPY
                ${MESHLIB_DATA_DIR}/resource
                ${MESHLIB_JSONS}
                ${MESHLIB_FONTS}
                resource
                ${LOCAL_JSONS}
            DESTINATION "${HTML_DIR}/assets"
        )
        mr_emscripten_pack_directory("${HTML_DIR}/assets" "/")

        # enable Asyncify
        mr_emscripten_set_async_func_list("${MESHLIB_DATA_DIR}/wasm/wasm_async_func_list.txt")
    endif()
endif()
