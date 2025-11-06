cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(cpp-examples CXX)

set(EXAMPLES
  # code examples
  Collision
  CollisionPrecise
  CollisionSelf
  ContourTriangulation
  FreeFormDeformation
  Fusion
  GlobalRegistration
  LaplacianDeformation
  MeshBoolean
  MeshDecimate
  MeshExport
  MeshExtrude
  MeshFillHole
  MeshFixDegeneracies
  MeshFromText
  MeshICP
  MeshLoadSave
  MeshModification
  MeshOffset
  MeshOffsetWeighted
  MeshStitchHole
  NoiseDenoise
  PointsToMesh
  SignedDistance
  SignedDistancePointToMesh
  SignedDistances
  TerrainTriangulation
  Triangulation
  # Viewer app example
  MRBasedMainFunc
)
foreach(TARGET ${EXAMPLES})
  add_executable(${TARGET} ${TARGET}.dox.cpp)
endforeach()

if(DEFINED MESHLIB_INCLUDE_DIRS AND DEFINED MESHLIB_LIB_DIRS)
  # for distributions without CMake config files
  foreach(TARGET ${EXAMPLES})
    target_include_directories(${TARGET} PUBLIC ${MESHLIB_INCLUDE_DIRS})
    target_link_directories(${TARGET} PUBLIC ${MESHLIB_LIB_DIRS})
    target_link_libraries(${TARGET} PRIVATE MRMesh)
  endforeach()
  # MeshFromText additionally depends on MRSymbolMesh
  target_link_libraries(MeshFromText PRIVATE MRSymbolMesh)
  # MeshOffset and Triangulation additionally depend on MRVoxels
  foreach(TARGET Fusion MeshOffset MeshOffsetWeighted Triangulation)
    target_link_libraries(${TARGET} PRIVATE MRVoxels)
  endforeach()
  # MeshLoadSave additionally depends on MRIOExtras
  target_link_libraries(MeshLoadSave PRIVATE MRIOExtras)
  # MRBasedMainFunc additionally depends on MRViewer
  target_link_libraries(MRBasedMainFunc PRIVATE MRViewer)
else()
  find_package(MeshLib CONFIG REQUIRED)
  find_package(TBB REQUIRED)
  find_package(Threads)
  
  foreach(TARGET ${EXAMPLES})
    target_include_directories(${TARGET} PUBLIC ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
    target_link_directories(${TARGET} PUBLIC ${MESHLIB_THIRDPARTY_LIB_DIR})
    target_link_libraries(${TARGET} PRIVATE MeshLib::MRMesh TBB::tbb)
  endforeach()
  # MeshFromText additionally depends on MRSymbolMesh
  target_link_libraries(MeshFromText PRIVATE MeshLib::MRSymbolMesh)
  # MeshOffset and Triangulation additionally depend on MRVoxels
  foreach(TARGET Fusion MeshOffset MeshOffsetWeighted Triangulation)
    target_link_libraries(${TARGET} PRIVATE MeshLib::MRVoxels)
  endforeach()
  # MeshLoadSave additionally depends on MRIOExtras
  target_link_libraries(MeshLoadSave PRIVATE MeshLib::MRIOExtras)
  # MRBasedMainFunc additionally depends on MRViewer
  target_link_libraries(MRBasedMainFunc PRIVATE MeshLib::MRViewer)
  if(APPLE)
    find_package(Boost CONFIG REQUIRED)
    target_link_libraries(MRBasedMainFunc PRIVATE Boost::headers)
  endif()
  
  if(EMSCRIPTEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MESHLIB_EMSCRIPTEN_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MESHLIB_EMSCRIPTEN_EXE_LINKER_FLAGS}")
  endif()
endif()
