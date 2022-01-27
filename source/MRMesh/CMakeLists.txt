cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (MRMesh CXX)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h")

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

#TODO: add relative system path to cholmod includes
target_include_directories(
  ${PROJECT_NAME} PRIVATE /usr/include/suitesparse/
)

find_package(GDCM CONFIG REQUIRED)
find_package(Freetype REQUIRED)
include_directories(${FREETYPE_INCLUDE_DIRS})
find_package(Boost COMPONENTS REQUIRED )
include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR})
pkg_check_modules(JSONCPP jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})

set(EIGEN_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/eigen)
set(PHMAP_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/parallel-hashmap)
target_include_directories(
	${PROJECT_NAME} PUBLIC ${EIGEN_DIR} ${PHMAP_DIR})

IF(APPLE)
	#jpeg-turbo is keg-only, which means it was not symlinked into /opt/homebrew,
	#because libjpeg-turbo is not linked to prevent conflicts with the standard libjpeg.
	#include_directories(/opt/homebrew/opt/jpeg-turbo/include)
	#link_directories(/opt/homebrew/opt/jpeg-turbo/lib)

	target_include_directories(
		${PROJECT_NAME} PUBLIC
		${MESHLIB_THIRDPARTY_INCLUDE_DIR}/expected/include
		${HOMEBREW_PREFIX}/opt/jpeg-turbo/include)

	target_link_libraries(${PROJECT_NAME} PRIVATE
		OpenVDB
		OpenCTM
		spdlog
		gtest gtest_main
		zip
		freetype
		pthread
		gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF
		Boost::boost
		jsoncpp
		cholmod
		cpr
		${PYTHON_LIB}
		podofo
		${HOMEBREW_PREFIX}/opt/jpeg-turbo/lib/libturbojpeg.dylib
		png
		fmt
		tbb
		)
ELSE()
	find_package(tl-expected REQUIRED)
	set(CPR_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/cpr)
	include_directories( ${CPR_DIR}/include)

	set(CURL_INCLUDES ${THIRDPARTY_LIB_DIR}/_deps/curl-src/include/)
	include_directories(${CURL_INCLUDES})

	target_link_libraries(${PROJECT_NAME} PRIVATE
		openvdb
		OpenCTM tl::expected
		spdlog
		gtest gtest_main
		zip
		freetype
		pthread
		gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF
		Boost::boost
		jsoncpp
		cholmod
		cpr
		${PYTHON_LIB}
		podofo
		turbojpeg
		)
ENDIF()


