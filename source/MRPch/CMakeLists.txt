project(MRPch CXX)

option(MR_PCH_USE_EXTRA_HEADERS "Add frequently used MeshLib headers to the precompiled header" OFF)

file(GLOB HEADERS "*.h")

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

IF(MR_PCH)
  add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

  IF(EMSCRIPTEN)
    target_compile_definitions(${PROJECT_NAME} PUBLIC
      EIGEN_STACK_ALLOCATION_LIMIT=0
      SPDLOG_FMT_EXTERNAL
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")

    target_link_libraries(${PROJECT_NAME} PRIVATE
      fmt
      jsoncpp
      tbb
    )
  ELSE()
    target_link_libraries(${PROJECT_NAME} PRIVATE
      Boost::boost
      Eigen3::Eigen
      fmt::fmt
      JsonCpp::JsonCpp
      spdlog::spdlog
      TBB::tbb
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
    target_compile_definitions(${PROJECT_NAME} PUBLIC _GNU_SOURCE)
  ENDIF()
ENDIF()
