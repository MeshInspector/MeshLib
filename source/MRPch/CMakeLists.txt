cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF(APPLE)
  project(MRPch CXX OBJCXX)
ELSE()
  project(MRPch CXX)
ENDIF()

IF(NOT MR_EMSCRIPTEN)
  find_package(Boost COMPONENTS REQUIRED)
  find_package(TBB REQUIRED)
  find_package(GDCM CONFIG REQUIRED)
  pkg_check_modules(JSONCPP jsoncpp)
  include_directories(${JSONCPP_INCLUDE_DIRS})
  find_package(Freetype REQUIRED)
  include_directories(${FREETYPE_INCLUDE_DIRS})
ELSE()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1 -s USE_FREETYPE=1 -s USE_LIBPNG=1")
ENDIF()

file(GLOB HEADERS "*.h")

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

IF(APPLE OR MR_EMSCRIPTEN)
  IF(APPLE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE _GNU_SOURCE)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        spdlog
        zip
        Boost::boost
        jsoncpp
        fmt
        tbb
        freetype
        gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF
        cpr
        ${MR_PYTHON_LIB}
    )
  ELSE()
    target_link_libraries(${PROJECT_NAME} PRIVATE
      jsoncpp
      tbb
    )
  ENDIF()
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    fmt
    spdlog
    Boost::boost
    jsoncpp
    tbb
    gdcmIOD gdcmDICT gdcmDSED gdcmMEXD gdcmMSFF
    cpr
    ${MR_PYTHON_LIB}
  )
ENDIF()

TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRPch.h>"
)
