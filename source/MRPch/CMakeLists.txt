project(MRPch CXX)

option(MR_PCH_USE_EXTRA_HEADERS "Add frequently used MeshLib headers to the precompiled header" OFF)

file(GLOB HEADERS "*.h")

install(
  FILES ${HEADERS}
  DESTINATION "${MR_INCLUDE_DIR}/${PROJECT_NAME}"
)

IF(MR_PCH)
  add_library(${PROJECT_NAME} STATIC "MRPch.cpp" ${HEADERS})

  find_package(tl-expected REQUIRED)
  target_link_libraries(${PROJECT_NAME} PRIVATE tl::expected)
  # TODO: CMake config
  target_include_directories(${PROJECT_NAME}
    PRIVATE
    $<BUILD_INTERFACE:${MESHLIB_THIRDPARTY_DIR}/parallel-hashmap>
  )

  # MRPch.h includes MRVoxels/MROpenVDB.h. Keep OpenVDB in the precompiled header only on MSVC, where
  # MRMesh/MRViewer build their own PCH and the compiler does not emit unreferenced inline functions, so
  # the many libraries that REUSE_FROM this shared PCH without linking libopenvdb still link cleanly.
  # On every other compiler such emission produces undefined OpenVDB symbols (e.g. openvdb::math::simplify
  # from inline ScaleMap methods), so exclude OpenVDB from the shared PCH there.
  IF(NOT MSVC OR NOT MESHLIB_BUILD_VOXELS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MRPCH_NO_OPENVDB)
  ENDIF()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # the full PCH slows the build down for GCC (it loads the whole PCH image in every TU)
    target_precompile_headers(${PROJECT_NAME}
      PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRStdlib.h>"
        "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRExpected.h>"
        "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/MRHashMap.h>"
    )
    return()
  endif()

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

  # On MSVC the shared PCH keeps OpenVDB (see MRPCH_NO_OPENVDB above), so it must be compiled with the
  # same OpenVDB definitions (OPENVDB_USE_DELAYED_LOADING, OPENVDB_ABI_VERSION_NUMBER) as the TUs that
  # reuse it; otherwise openvdb/version.h bakes its own defaults into the PCH and consumers that link
  # OpenVDB hit a macro-redefinition. Linking the imported target propagates those defs.
  IF(MSVC AND MESHLIB_BUILD_VOXELS)
    find_package(OpenVDB 10 REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenVDB::openvdb)
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
