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

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR})
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

		IF(NOT APPLE)
			set(CPR_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/cpr)
			target_include_directories(${PROJECT_NAME} PUBLIC ${CPR_DIR}/include)
			target_include_directories(${PROJECT_NAME} PUBLIC ${MESHLIB_THIRDPARTY_LIB_DIR}/cpr/cpr_generated_includes)

			set(CURL_INCLUDES ${THIRDPARTY_LIB_DIR}/_deps/curl-src/include/)
			target_include_directories(${PROJECT_NAME} PUBLIC ${CURL_INCLUDES})
		ENDIF()
	ENDIF()

	IF(MRMESH_NO_PYTHON)
		target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_PYTHON)
	ELSE()
		list(APPEND MRMESH_OPTIONAL_DEPENDENCIES ${PYTHON_LIB})
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
ELSE()
	target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_PDF)
	target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_CPR)
	target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_PYTHON)
	target_compile_definitions(${PROJECT_NAME} PRIVATE MRMESH_NO_DICOM)
ENDIF()

IF(NOT MR_EMSCRIPTEN)
  find_package(Boost COMPONENTS REQUIRED )
  find_package( TBB REQUIRED )
  pkg_check_modules(JSONCPP jsoncpp)
  include_directories(${JSONCPP_INCLUDE_DIRS})
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1 -s USE_FREETYPE=1 -s USE_LIBPNG=1")
ENDIF()

set(EIGEN_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/eigen)
set(PHMAP_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/parallel-hashmap)
target_include_directories(
	${PROJECT_NAME} PUBLIC ${EIGEN_DIR} ${PHMAP_DIR})

IF(APPLE OR MR_EMSCRIPTEN)
	#jpeg-turbo is keg-only, which means it was not symlinked into /opt/homebrew,
	#because libjpeg-turbo is not linked to prevent conflicts with the standard libjpeg.
	#include_directories(/opt/homebrew/opt/jpeg-turbo/include)
	#link_directories(/opt/homebrew/opt/jpeg-turbo/lib)

	target_include_directories(
		${PROJECT_NAME} PUBLIC
		${MESHLIB_THIRDPARTY_INCLUDE_DIR}/expected/include
		${HOMEBREW_PREFIX}/opt/jpeg-turbo/include)

	IF (NOT MR_EMSCRIPTEN)
		find_library(COCOA_LIBRARY Cocoa ONLY)

		target_link_libraries(${PROJECT_NAME} PRIVATE
		    OpenVDB
		    spdlog
		    gtest gtest_main
		    zip
		    pthread
		    Boost::boost
		    jsoncpp
		    ${HOMEBREW_PREFIX}/opt/jpeg-turbo/lib/libturbojpeg.dylib
		    png
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
	target_include_directories(${PROJECT_NAME} PUBLIC ${MESHLIB_THIRDPARTY_LIB_DIR}/openvdb/openvdb/openvdb)
	target_include_directories(${PROJECT_NAME} PUBLIC ${MESHLIB_THIRDPARTY_LIB_DIR}/openvdb/openvdb/openvdb/openvdb)

	target_link_libraries(${PROJECT_NAME} PRIVATE
		openvdb
		tl::expected
		fmt
		spdlog
		gtest gtest_main
		zip
		pthread
		Boost::boost
		jsoncpp
		turbojpeg
		${MRMESH_OPTIONAL_DEPENDENCIES}
		)
ENDIF()
