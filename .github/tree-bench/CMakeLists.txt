# tree benchmark, added to a MeshLib checkout by tmp-tree-bench.yml and tmp-tree-bench-wasm.yml
find_package(Threads REQUIRED)
find_package(TBB 2020.1 COMPONENTS tbb REQUIRED)
add_executable(treebench treebench.cpp)
target_link_libraries(treebench PRIVATE MRMesh TBB::tbb)
if(EMSCRIPTEN)
  # run by node: the program exits when main returns; with threads, the workers are created in advance,
  # since a worker requested later starts only when the main thread returns to the event loop
  target_link_options(treebench PRIVATE "-sEXIT_RUNTIME=1")
  # the single-threaded build of MeshLib allows 2GB, too few for all the meshes of the benchmark with the trees of the PR
  target_link_options(treebench PRIVATE "-sMAXIMUM_MEMORY=4GB")
  if(NOT MR_EMSCRIPTEN_SINGLETHREAD)
    target_link_options(treebench PRIVATE "-sPTHREAD_POOL_SIZE=8")
  endif()
endif()
