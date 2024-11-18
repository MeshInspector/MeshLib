cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD ${MR_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(meshconv CXX)

find_package(Boost ${MR_MIN_BOOST_VERSION} COMPONENTS program_options REQUIRED)
IF(Boost_PROGRAM_OPTIONS_FOUND)
  link_libraries(${Boost_PROGRAM_OPTIONS_LIBRARY})
ENDIF()

add_executable(${PROJECT_NAME} meshconv.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
  MRMesh
  MRIOExtras
  fmt
  spdlog
  Boost::boost
  tbb
)

IF(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/../lib;@loader_path;@loader_path/../lib/thirdparty_lib;@loader_path/meshlib;${MESHLIB_THIRDPARTY_ROOT_DIR}/lib;${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_MACOSX_RPATH TRUE) # Ensure RPATH is properly set for macOS
ENDIF()


install(TARGETS ${PROJECT_NAME} DESTINATION "${MR_BIN_DIR}")

IF(MR_PCH)
  TARGET_PRECOMPILE_HEADERS(${PROJECT_NAME} REUSE_FROM MRPch)
ENDIF()
