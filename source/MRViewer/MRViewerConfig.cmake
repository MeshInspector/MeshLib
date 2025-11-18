include(CMakeFindDependencyMacro)

if(NOT EMSCRIPTEN)
  find_dependency(glfw3)
endif()

# static builds require to find private dependencies
if(EMSCRIPTEN)
  find_dependency(glad)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/MRViewerTargets.cmake")

check_required_components(MRViewer)
