cmake_minimum_required(VERSION 3.18)
project(region_crossovers CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(region_crossovers region_crossovers.cpp)

# same two ways to find MeshLib as examples/cpp-examples/CMakeLists.txt
if(DEFINED MESHLIB_INCLUDE_DIRS AND DEFINED MESHLIB_LIB_DIRS)
  target_include_directories(region_crossovers PRIVATE ${MESHLIB_INCLUDE_DIRS})
  target_link_directories(region_crossovers PRIVATE ${MESHLIB_LIB_DIRS})
  target_link_libraries(region_crossovers PRIVATE MRMesh)
elseif(DEFINED MRMESH_CONFIG_DIR)
  # only MRMesh's config, so other MeshLib components' dependencies (OpenVDB, ...) are not needed
  list(APPEND CMAKE_MODULE_PATH "${MRMESH_CONFIG_DIR}/Modules") # MeshLib's Find modules (phmap, ...), as meshlib-config does
  find_package(MRMesh CONFIG REQUIRED PATHS ${MRMESH_CONFIG_DIR} NO_DEFAULT_PATH)
  target_include_directories(region_crossovers PRIVATE ${MESHLIB_INCLUDE_DIR} ${MESHLIB_INCLUDE_DIR}/include)
  target_link_libraries(region_crossovers PRIVATE MeshLib::MRMesh)
else()
  find_package(MeshLib CONFIG REQUIRED)
  target_include_directories(region_crossovers PRIVATE ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
  target_link_directories(region_crossovers PRIVATE ${MESHLIB_THIRDPARTY_LIB_DIR})
  target_link_libraries(region_crossovers PRIVATE MeshLib::MRMesh)
endif()

if(MSVC)
  target_compile_options(region_crossovers PRIVATE /utf-8 /permissive-)
  target_compile_definitions(region_crossovers PRIVATE NOMINMAX _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR SPDLOG_COMPILED_LIB SPDLOG_SHARED_LIB)
endif()
