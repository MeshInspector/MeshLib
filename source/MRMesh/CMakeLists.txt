cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRMesh CXX)

option(MRMESH_NO_CLIPBOARD "Disable clipboard support for Linux and Mac" OFF)
option(MRMESH_NO_PDF "Disable PDF support" OFF)
option(MRMESH_NO_CPR "Disable CPR tests" OFF)
option(MRMESH_NO_PYTHON "Disable Python bindings" OFF)
option(MRMESH_NO_DICOM "Disable DICOM image support" OFF)
option(MRMESH_NO_OPENCTM "Disable OpenCTM support" OFF)
option(MRMESH_NO_LABEL "Disable support of label objects" OFF)
option(MRMESH_NO_JPEG "Disable JPEG support" OFF)
option(MRMESH_NO_PNG "Disable PNG support" OFF)
option(MRMESH_NO_TIFF "Disable TIFF support" OFF)
option(MRMESH_NO_VOXEL "Disable voxel support" OFF)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set(MRMESH_OPTIONAL_DEPENDENCIES "")

IF(MRMESH_NO_OPENCTM)
	target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_OPENCTM)
ELSE()
	list(APPEND MRMESH_OPTIONAL_DEPENDENCIES OpenCTM)
ENDIF()

IF(NOT MR_EMSCRIPTEN)
	IF(MRMESH_NO_CLIPBOARD)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_CLIPBOARD)
	ELSE()
		list(APPEND MRMESH_OPTIONAL_DEPENDENCIES clip)
	ENDIF()

	IF(MRMESH_NO_PDF)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_PDF)
	ELSE()
		list(APPEND MRMESH_OPTIONAL_DEPENDENCIES podofo)
	ENDIF()

	IF(MRMESH_NO_CPR)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_CPR)
	ELSE()
		list(APPEND MRMESH_OPTIONAL_DEPENDENCIES cpr)
	ENDIF()

	IF(MRMESH_NO_PYTHON)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_PYTHON)
	ELSE()
		list(APPEND MRMESH_OPTIONAL_DEPENDENCIES ${MR_PYTHON_LIB})
	ENDIF()

	IF(MRMESH_NO_DICOM)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_DICOM)
	ELSE()
		find_package(GDCM CONFIG REQUIRED)
		list(APPEND MRMESH_OPTIONAL_DEPENDENCIES gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF)
	ENDIF()

	IF(MRMESH_NO_LABEL)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_LABEL)
	ELSE()
		find_package(Freetype REQUIRED)
		include_directories(${FREETYPE_INCLUDE_DIRS})
		list(APPEND MRMESH_OPTIONAL_DEPENDENCIES freetype)
	ENDIF()

	IF(MRMESH_NO_JPEG)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_JPEG)
	ELSE()
		IF(APPLE)
			target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${HOMEBREW_PREFIX}/opt/jpeg-turbo/include>) # @todo
			list(APPEND MRMESH_OPTIONAL_DEPENDENCIES ${HOMEBREW_PREFIX}/opt/jpeg-turbo/lib/libturbojpeg.dylib)
		ELSE()
			list(APPEND MRMESH_OPTIONAL_DEPENDENCIES turbojpeg)
		ENDIF()
	ENDIF()

	IF(MRMESH_NO_PNG)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_PNG)
	ELSE()
		list(APPEND MRMESH_OPTIONAL_DEPENDENCIES png)
	ENDIF()

	IF(MRMESH_NO_TIFF)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_TIFF)
	ELSE()
		list(APPEND MRMESH_OPTIONAL_DEPENDENCIES tiff)
	ENDIF()


	IF(MRMESH_NO_VOXEL)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_VOXEL)
	ELSE()
		IF(APPLE)
			list(APPEND MRMESH_OPTIONAL_DEPENDENCIES OpenVDB)
		ELSE()
			list(APPEND MRMESH_OPTIONAL_DEPENDENCIES openvdb)
		ENDIF()
	ENDIF()
ELSE()
	target_compile_definitions(${PROJECT_NAME}
		PRIVATE
			MRMESH_NO_PDF
			MRMESH_NO_CPR
			MRMESH_NO_PYTHON
			MRMESH_NO_DICOM
			MRMESH_NO_JPEG
			MRMESH_NO_VOXEL
	)
ENDIF()

# alternative Eigen include for external project
find_package(Eigen3 NO_MODULE)
if(DEFINED EIGEN3_FOUND)
  target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIRS}>)
  list(APPEND MRMESH_OPTIONAL_DEPENDENCIES Eigen3::Eigen)
endif()

IF(NOT MR_EMSCRIPTEN)
  find_package(Boost COMPONENTS REQUIRED )
  find_package( TBB REQUIRED )
  pkg_check_modules(JSONCPP jsoncpp)
  include_directories(${JSONCPP_INCLUDE_DIRS})
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1 -s USE_FREETYPE=1 -s USE_LIBPNG=1")
ENDIF()

IF(APPLE OR MR_EMSCRIPTEN)
	#jpeg-turbo is keg-only, which means it was not symlinked into /opt/homebrew,
	#because libjpeg-turbo is not linked to prevent conflicts with the standard libjpeg.
	#include_directories(/opt/homebrew/opt/jpeg-turbo/include)
	#link_directories(/opt/homebrew/opt/jpeg-turbo/lib)

	IF (NOT MR_EMSCRIPTEN)
		find_library(COCOA_LIBRARY Cocoa ONLY)
	    target_compile_definitions(${PROJECT_NAME} PRIVATE _GNU_SOURCE)
		target_link_libraries(${PROJECT_NAME} PRIVATE
		    spdlog
		    gtest gtest_main
		    zip
		    pthread
		    Boost::boost
		    jsoncpp
		    fmt
		    tbb
			${MRMESH_OPTIONAL_DEPENDENCIES}
			${COCOA_LIBRARY}
		)
	ELSE()
	    target_link_libraries(${PROJECT_NAME} PRIVATE
	        gtest gtest_main
	        zip
	        freetype
	        pthread
	        jsoncpp
	        tbb
			${MRMESH_OPTIONAL_DEPENDENCIES}
	    )
	ENDIF()
ELSE()
	find_package(tl-expected REQUIRED)
	target_link_libraries(${PROJECT_NAME} PRIVATE
		tl::expected
		fmt
		spdlog
		gtest gtest_main
		zip
		pthread
		Boost::boost
		jsoncpp
		tbb
		${MRMESH_OPTIONAL_DEPENDENCIES}
		)
ENDIF()

set(CMAKE_INSTALL_PREFIX "/Library/Frameworks")
install(TARGETS ${PROJECT_NAME}
		EXPORT ${PROJECT_NAME}
		DESTINATION "${MR_MAIN_LIB_DIR}")

install(FILES ${HEADERS}
		DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake
		DESTINATION ${MR_CONFIG_DIR}/cmake)


install(EXPORT ${PROJECT_NAME}
		FILE ${PROJECT_NAME}Targets.cmake
		NAMESPACE MeshLib::
		DESTINATION ${MR_CONFIG_DIR}/cmake)

