project(MRPch CXX)

option(MR_PCH_USE_EXTRA_HEADERS "Add frequently used MeshLib headers to the precompiled header" OFF)

file(GLOB HEADERS "*.h")

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

IF(MR_PCH)
  add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

  # The lean PCH (see MR_PCH_LEAN in MRPch.h) contains no third-party headers
  # except parallel-hashmap and tl::expected, so it needs none of these
  # dependencies — and must not inherit their interface compile definitions
  # (e.g. FMT_SHARED), which GCC would require in every TU that uses the PCH.
  IF(NOT MR_PCH_LEAN)
    # Emscripten's Boost package doesn't contain CMake config files
    # on other systems the module mode shows a warning
    IF(NOT EMSCRIPTEN)
      find_package(Boost 1.73 CONFIG COMPONENTS headers REQUIRED)
    ELSE()
      find_package(Boost 1.73 REQUIRED)
    ENDIF()
    find_package(Eigen3 REQUIRED)
    find_package(fmt 8 REQUIRED)
    find_package(JsonCpp REQUIRED)
    find_package(spdlog 1.9 REQUIRED)
    find_package(TBB 2020.1 COMPONENTS tbb REQUIRED)

    target_link_libraries(${PROJECT_NAME} PRIVATE
      Boost::boost
      Eigen3::Eigen
      fmt::fmt
      JsonCpp::JsonCpp
      spdlog::spdlog
      TBB::tbb
    )

    # GTest module defines a macro for MRMesh and PCH must match it (but not for MSVC)
    IF(BUILD_TESTING AND NOT MRMESH_NO_GTEST AND NOT MSVC)
      find_package(GTest REQUIRED)
      target_link_libraries(${PROJECT_NAME} PRIVATE GTest::gtest)
    ENDIF()
  ENDIF()

  # TODO: CMake config
  target_include_directories(${PROJECT_NAME}
    PRIVATE
      $<BUILD_INTERFACE:${MESHLIB_THIRDPARTY_DIR}/parallel-hashmap>
  )

  IF(EMSCRIPTEN)
    target_compile_definitions(${PROJECT_NAME} PUBLIC
      EIGEN_STACK_ALLOCATION_LIMIT=0
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_BOOST_HEADERS=1")
    # FIXME: the option is required for MeshInspector but not for MeshLib
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
  ENDIF()

  IF(MR_PCH_USE_EXTRA_HEADERS AND NOT MR_PCH_LEAN)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MR_PCH_USE_EXTRA_HEADERS)
  ENDIF()
  IF(NOT MESHLIB_BUILD_MRVIEWER AND NOT MR_PCH_LEAN)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MESHLIB_NO_VIEWER)
  ENDIF()
  IF(MR_PCH_USE_EXTRA_HEADERS AND MESHLIB_BUILD_MRVIEWER AND NOT MR_PCH_LEAN)
    target_link_libraries(${PROJECT_NAME} PUBLIC imgui)
  ENDIF()

  target_precompile_headers(${PROJECT_NAME} PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRPch.h>"
  )

  # `dladdr` in our patched `PYBIND11_MODULE(...)` needs this.
  # Also needed for `Boost.Stacktrace` apparently.
  # Not in lean mode: the lean PCH needs neither, and on Linux the consuming
  # targets don't define _GNU_SOURCE, which would invalidate the PCH for GCC.
  IF(NOT WIN32 AND NOT MR_PCH_LEAN)
    target_compile_definitions(${PROJECT_NAME} PUBLIC _GNU_SOURCE)
  ENDIF()
ENDIF()
