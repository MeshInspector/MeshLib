# tree benchmark, added to a MeshLib checkout by tmp-tree-bench.yml
find_package(TBB 2020.1 COMPONENTS tbb REQUIRED)
add_executable(treebench treebench.cpp)
target_link_libraries(treebench PRIVATE MRMesh TBB::tbb)
