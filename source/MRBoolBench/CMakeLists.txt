project(MRBoolBench CXX)

# throw-away benchmark: the same Boolean workload with shared mimalloc, static mimalloc.o and glibc malloc
add_executable(MRBoolBenchShared MRBoolBench.cpp)
add_executable(MRBoolBenchStatic MRBoolBench.cpp)
add_executable(MRBoolBenchGlibc MRBoolBench.cpp)
foreach(t MRBoolBenchShared MRBoolBenchStatic MRBoolBenchGlibc)
  target_link_libraries(${t} PRIVATE MRMesh ${CMAKE_DL_LIBS})
endforeach()

include(Mimalloc)
mr_enable_mimalloc(MRBoolBenchShared)

# master's Linux mr_enable_mimalloc
set(_mimalloc_dir ${CMAKE_BINARY_DIR}/mimalloc_static)
include(ExternalProject)
ExternalProject_Add(mimalloc_static_build
  URL https://github.com/microsoft/mimalloc/archive/refs/tags/v3.4.3.tar.gz
  URL_HASH SHA256=7b043c26b64dde66344d144fbcac3664858a4eb32197793ea2a18feb32741ca1
  INSTALL_DIR ${_mimalloc_dir}
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_LIBDIR=lib
    -DMI_INSTALL_TOPLEVEL=ON
    -DMI_USE_CXX=ON
    -DMI_BUILD_SHARED=OFF
    -DMI_BUILD_TESTS=OFF
  BUILD_BYPRODUCTS ${_mimalloc_dir}/lib/mimalloc.o
)
add_dependencies(MRBoolBenchStatic mimalloc_static_build)
set_source_files_properties(${_mimalloc_dir}/lib/mimalloc.o PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_sources(MRBoolBenchStatic PRIVATE ${_mimalloc_dir}/lib/mimalloc.o)
