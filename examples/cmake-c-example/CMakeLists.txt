cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(cmake-c-examples C)

find_package(MeshLib CONFIG REQUIRED)

add_executable(MeshBoolean MeshBoolean.c)
add_executable(MeshDecimate MeshDecimate.c)
add_executable(MeshFillHole MeshFillHole.c)
add_executable(MeshImportExport MeshImportExport.c)
add_executable(MeshOffset MeshOffset.c)
add_executable(MeshResolveDegenerations MeshResolveDegenerations.c)

foreach(TARGET MeshBoolean MeshDecimate MeshFillHole MeshImportExport MeshOffset MeshResolveDegenerations)
  target_include_directories(${TARGET} PUBLIC ${MESHLIB_INCLUDE_DIR} ${MESHLIB_THIRDPARTY_INCLUDE_DIR})
  target_link_libraries(${TARGET} PRIVATE MeshLib::MRMeshC)
endforeach()
