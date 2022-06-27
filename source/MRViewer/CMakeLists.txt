cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (MRViewer CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

IF(MR_EMSCRIPTEN)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")
ENDIF() # MR_EMSCRIPTEN

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

IF(NOT MR_EMSCRIPTEN)
if(GTKMM_FOUND)
  target_include_directories( ${PROJECT_NAME} PUBLIC ${GTKMM_INCLUDE_DIRS})
endif()

set(GLAD_INCLUDES ${THIRDPARTY_LIB_DIR}/glad/include/)
target_include_directories( ${PROJECT_NAME} PUBLIC ${GLAD_INCLUDES})
ENDIF() # NOT MR_EMSCRIPTEN

IF(MR_EMSCRIPTEN)
	target_link_libraries(${PROJECT_NAME} PRIVATE
			MRMesh
			tbb
			jsoncpp
			imgui
			glfw
			)
ELSE() # MR_EMSCRIPTEN
IF(APPLE)
	target_link_libraries(${PROJECT_NAME} PRIVATE
			MRMesh
			fmt
			tbb
			OpenVDB
			spdlog
			jsoncpp
			cpr
			sigc-3.0
			glib-2.0
			glibmm-2.4
			imgui
			glad
			glfw
			gtkmm-3.0
			)
ELSE() # APPLE
	target_link_libraries(${PROJECT_NAME} PRIVATE
			MRMesh
			imgui
			glad
			glfw
			gtkmm-3.0
			)
ENDIF() # APPLE
ENDIF() # MR_EMSCRIPTEN

file(GLOB AWESOME_FONTS "${THIRDPARTY_INCLUDE_DIR}/fontawesome-free/*.ttf")
file(GLOB IMGUI_FONTS "${THIRDPARTY_INCLUDE_DIR}/imgui/misc/fonts/*.ttf")
IF(MR_EMSCRIPTEN)
	file(COPY ${AWESOME_FONTS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
	file(COPY ${IMGUI_FONTS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
	file(COPY "resource" DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
ELSE() # MR_EMSCRIPTEN
	file(COPY ${AWESOME_FONTS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	file(COPY ${IMGUI_FONTS} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	file(COPY "resource" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
ENDIF() # MR_EMSCRIPTEN
