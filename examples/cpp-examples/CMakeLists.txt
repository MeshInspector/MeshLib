cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cpp-examples CXX)

find_package(MeshLib CONFIG REQUIRED)
find_package(TBB REQUIRED)

set(EXAMPLES
  MeshBoolean
  MeshDecimate
  MeshExport
  MeshExtrude
  MeshFillHole
  MeshICP
  MeshLoadSave
  MeshModification
  MeshOffset
  MeshStitchHole
)

foreach(TARGET ${EXAMPLES})
  add_executable(${TARGET} ${TARGET}.dox.cpp)
  target_include_directories(${TARGET} PUBLIC ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
  target_link_libraries(${TARGET} PRIVATE MeshLib::MRMesh TBB::tbb)
endforeach()

add_executable(MRBasedMainFunc MRBasedMainFunc.dox.cpp)
target_include_directories(MRBasedMainFunc PUBLIC ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
target_link_libraries(MRBasedMainFunc PRIVATE MeshLib::MRMesh MeshLib::MRViewer)
