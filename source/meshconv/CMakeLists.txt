cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(meshconv CXX)

find_package(Boost COMPONENTS program_options REQUIRED)

add_executable(${PROJECT_NAME} meshconv.cpp)

IF(WIN32 OR MESHLIB_USE_VCPKG)
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    MRIOExtras
    fmt::fmt
    spdlog::spdlog
    Boost::program_options
    TBB::tbb
  )
ELSE()
  target_link_libraries(${PROJECT_NAME} PRIVATE
    MRMesh
    MRIOExtras
    fmt
    spdlog
    Boost::program_options
    tbb
  )
ENDIF()

install(TARGETS ${PROJECT_NAME} DESTINATION "${MR_BIN_DIR}")

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
