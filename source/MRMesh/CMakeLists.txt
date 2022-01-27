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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1 -s USE_FREETYPE=1")
#add_compile_options(-s USE_BOOST_HEADERS=1)

#find_package(GDCM CONFIG REQUIRED)
#find_package(Freetype REQUIRED)
#include_directories(${FREETYPE_INCLUDE_DIRS})
#find_package(Boost COMPONENTS REQUIRED )
#find_package( TBB REQUIRED )
add_compile_definitions(__emscripten__)
include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR})
include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/wasmtbb/include)
include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/spdlog/include)
include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/googletest/googletest/include)
include_directories(${MESHLIB_THIRDPARTY_INCLUDE_DIR}/libzip/lib)
include_directories(${THIRDPARTY_LIB_DIR}/libzip)
pkg_check_modules(JSONCPP jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})

set(EIGEN_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/eigen)
set(PHMAP_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/parallel-hashmap)
target_include_directories(
	${PROJECT_NAME} PUBLIC ${EIGEN_DIR} ${PHMAP_DIR})

#IF(APPLE)
	target_include_directories(
		${PROJECT_NAME} PUBLIC
		${MESHLIB_THIRDPARTY_INCLUDE_DIR}/expected/include)

	target_link_libraries(${PROJECT_NAME} PRIVATE
		#OpenVDB
		OpenCTM
		spdlog
		gtest gtest_main
		zip
		freetype
		pthread
		#gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF
		#Boost::boost
		jsoncpp
		#cholmod
		#cpr
		#${PYTHON_LIB}
		#podofo
		#turbojpeg
		png
		fmt
		tbb
		)
#ELSE()
#	find_package(tl-expected REQUIRED)
#	set(CPR_DIR ${MESHLIB_THIRDPARTY_INCLUDE_DIR}/cpr)
#	include_directories( ${CPR_DIR}/include)
#
#	set(CURL_INCLUDES ${THIRDPARTY_LIB_DIR}/_deps/curl-src/include/)
#	include_directories(${CURL_INCLUDES})
#
#	target_link_libraries(${PROJECT_NAME} PRIVATE
#		openvdb
#		OpenCTM tl::expected
#		spdlog
#		gtest gtest_main
#		zip
#		freetype
#		pthread
#		gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF
#		Boost::boost
#		jsoncpp
#		cholmod
#		cpr
#		${PYTHON_LIB}
#		podofo
#		turbojpeg
#		)
#ENDIF()


