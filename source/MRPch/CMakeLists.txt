cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRPch CXX)

file(GLOB HEADERS "*.h")

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

# for not Emscripten builds PCH does not give any speedup, and reveals a lot of compiler bugs
IF(MR_EMSCRIPTEN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1 -s USE_FREETYPE=1 -s USE_LIBPNG=1")

  add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

  target_link_libraries(${PROJECT_NAME} PRIVATE
    jsoncpp
    tbb
  )

  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRPch.h>"
  )
ENDIF()
