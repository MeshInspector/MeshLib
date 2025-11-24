project(MRPch CXX)

option(MR_PCH_USE_EXTRA_HEADERS "Add frequently used MeshLib headers to the precompiled header" OFF)

file(GLOB HEADERS "*.h")

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

IF(MR_PCH)
  add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

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
