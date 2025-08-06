cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MRPch CXX)

option(MR_PCH_USE_EXTRA_HEADERS "Add frequently used MeshLib headers to the precompiled header" OFF)

file(GLOB HEADERS "*.h")

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

IF(MR_PCH)
  IF(MR_EMSCRIPTEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")

    add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

    target_compile_definitions(${PROJECT_NAME} PUBLIC SPDLOG_FMT_EXTERNAL)
    # FIXME: comment required
    target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_STACK_ALLOCATION_LIMIT=0)

    target_link_libraries(${PROJECT_NAME} PRIVATE
      fmt
      jsoncpp
      tbb
    )
  ELSEIF(WIN32 OR MESHLIB_USE_VCPKG)
    find_package(Boost COMPONENTS REQUIRED)

    add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

    target_link_libraries(${PROJECT_NAME} PRIVATE
      fmt::fmt
      spdlog::spdlog
      Boost::boost
      JsonCpp::JsonCpp
      TBB::tbb
    )
  ELSE()
    find_package(Boost COMPONENTS REQUIRED)
    include_directories(${jsoncpp_INCLUDE_DIRS})

    add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

    target_link_libraries(${PROJECT_NAME} PRIVATE
      fmt
      spdlog
      Boost::boost
      JsonCpp::JsonCpp
      tbb
    )
  ENDIF()

  IF(MR_PCH_USE_EXTRA_HEADERS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MR_PCH_USE_EXTRA_HEADERS)
  ENDIF()
  IF(NOT MESHLIB_BUILD_MRVIEWER)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VIEWER)
  ENDIF()
  IF(MR_PCH_USE_EXTRA_HEADERS AND MESHLIB_BUILD_MRVIEWER)
    target_link_libraries(${PROJECT_NAME} PUBLIC imgui)
  ENDIF()

  target_precompile_headers(${PROJECT_NAME} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRPch.h>"
  )

  # `dladdr` in our patched `PYBIND11_MODULE(...)` needs this.
  # Also needed for `Boost.Stacktrace` apparently.
  IF(NOT WIN32)
    target_compile_definitions(${PROJECT_NAME} PUBLIC
      _GNU_SOURCE
    )
  ENDIF()
ENDIF()
