cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (MeshViewer CXX)

IF(NOT MR_EMSCRIPTEN)
IF(APPLE)
	add_executable(${PROJECT_NAME} MACOSX_BUNDLE MRViewerApp.cpp)
ELSE()
	add_executable(${PROJECT_NAME} MRViewerApp.cpp)
ENDIF()
ELSE() # NOT MR_EMSCRIPTEN
	add_executable(${PROJECT_NAME} MRViewerApp.cpp)
ENDIF() # NOT MR_EMSCRIPTEN

file(GLOB LIB_LIST "../*.json")
file(GLOB PNGS "*.png")
IF(MR_EMSCRIPTEN)
	file(GLOB LOCAL_WASM_FILES "../../wasm/*.*")
	file(COPY ${LOCAL_WASM_FILES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
ENDIF() # MR_EMSCRIPTEN
file(COPY ${LIB_LIST} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${PNGS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

IF(MR_EMSCRIPTEN)
target_link_libraries(${PROJECT_NAME} PRIVATE
    pthread
	zip
	gtest
	-Wl,--whole-archive
	MRCommonPlugins
	MRMesh
	MRViewer
	-Wl,--no-whole-archive
)
ELSE() # MR_EMSCRIPTEN
target_link_libraries(${PROJECT_NAME} PRIVATE
	MRViewer
	MRMesh
)
ENDIF() # MR_EMSCRIPTEN

install(TARGETS ${PROJECT_NAME} DESTINATION "${main_lib_dest}")
install(FILES ${HEADERS} DESTINATION "${include_dest}/${PROJECT_NAME}")
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME} DESTINATION "${lib_dest}")