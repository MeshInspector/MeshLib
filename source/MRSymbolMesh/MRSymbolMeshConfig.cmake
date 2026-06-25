include(CMakeFindDependencyMacro)

# static builds require to find private dependencies
if(EMSCRIPTEN)
  find_dependency(Freetype)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/MRSymbolMeshTargets.cmake")

check_required_components(MRSymbolMesh)
