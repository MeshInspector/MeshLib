include(CMakeFindDependencyMacro)

# static builds require to find private dependencies
if(EMSCRIPTEN)
  find_dependency(Freetype)
  find_dependency(phmap 2.0)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/imguiTargets.cmake")

check_required_components(imgui)
