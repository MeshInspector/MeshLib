cmake_minimum_required(VERSION 3.18)
project(ml_bench CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ml_bench ml_bench.cpp)

# Same two ways to find MeshLib as MeshLib's examples/cpp-examples/CMakeLists.txt
if(DEFINED MESHLIB_INCLUDE_DIRS AND DEFINED MESHLIB_LIB_DIRS)
  # a source checkout built with MSBuild, or a distribution without CMake config files
  target_include_directories(ml_bench PRIVATE ${MESHLIB_INCLUDE_DIRS})
  target_link_directories(ml_bench PRIVATE ${MESHLIB_LIB_DIRS})
  target_link_libraries(ml_bench PRIVATE MRMesh)
elseif(DEFINED MRMESH_CONFIG_DIR)
  # only MRMesh's config from an unpacked package, so other MeshLib components' dependencies are not needed
  list(APPEND CMAKE_MODULE_PATH "${MRMESH_CONFIG_DIR}/Modules")
  find_package(MRMesh CONFIG REQUIRED PATHS ${MRMESH_CONFIG_DIR} NO_DEFAULT_PATH)
  target_include_directories(ml_bench PRIVATE ${MESHLIB_INCLUDE_DIR} ${MESHLIB_INCLUDE_DIR}/include)
  target_link_libraries(ml_bench PRIVATE MeshLib::MRMesh)
else()
  find_package(MeshLib CONFIG REQUIRED)
  target_include_directories(ml_bench PRIVATE ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
  target_link_directories(ml_bench PRIVATE ${MESHLIB_THIRDPARTY_LIB_DIR})
  target_link_libraries(ml_bench PRIVATE MeshLib::MRMesh)
endif()

if(MSVC)
  target_compile_options(ml_bench PRIVATE /utf-8 /permissive-)
  target_compile_definitions(ml_bench PRIVATE NOMINMAX _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR SPDLOG_COMPILED_LIB SPDLOG_SHARED_LIB)
endif()
