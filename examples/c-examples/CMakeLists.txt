cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(cmake-c-examples C)

set(EXAMPLES
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
)

foreach(TARGET ${EXAMPLES})
  add_executable(${TARGET} ${TARGET}.dox.c)
endforeach()

if(DEFINED MESHLIB_INCLUDE_DIRS AND DEFINED MESHLIB_LIB_DIRS)
  # for distributions without CMake config files
  foreach(TARGET ${EXAMPLES})
    target_include_directories(${TARGET} PUBLIC ${MESHLIB_INCLUDE_DIRS})
    target_link_directories(${TARGET} PUBLIC ${MESHLIB_LIB_DIRS})
    target_link_libraries(${TARGET} PRIVATE MeshLibC2)
  endforeach()
else()
  find_package(MeshLib CONFIG REQUIRED)
  
  foreach(TARGET ${EXAMPLES})
    target_include_directories(${TARGET} PUBLIC ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
    target_link_directories(${TARGET} PRIVATE ${MESHLIB_THIRDPARTY_LIB_DIR})
    target_link_libraries(${TARGET} PRIVATE MeshLib::MeshLibC2)
  endforeach()
endif()

if(NOT MSVC)
  # Triangulation requires the math library
  target_link_libraries(Triangulation PRIVATE m)
endif()
