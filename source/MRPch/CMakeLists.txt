cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRPch CXX)

file(GLOB HEADERS "*.h")

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

IF(MR_PCH)
  IF(MR_EMSCRIPTEN)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1 -s USE_FREETYPE=1 -s USE_LIBPNG=1")

    add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

    target_link_libraries(${PROJECT_NAME} PRIVATE
      jsoncpp
      tbb
    )

  ELSE()

    find_package(Boost COMPONENTS REQUIRED)
    find_package(TBB REQUIRED)
    find_package(GDCM CONFIG REQUIRED)
    pkg_check_modules(JSONCPP jsoncpp)
    include_directories(${JSONCPP_INCLUDE_DIRS})

    add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

    target_link_libraries(${PROJECT_NAME} PRIVATE
      fmt
      spdlog
      Boost::boost
      jsoncpp
      tbb
      gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF
      cpr
      Python::Python
    )
  ENDIF()

  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRPch.h>"
  )

ENDIF()
