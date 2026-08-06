include(CMakeFindDependencyMacro)

# static builds require to find private dependencies
if(EMSCRIPTEN)
  find_dependency(OpenCTM)
  find_dependency(LAZPERF)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/MRIOExtrasTargets.cmake")
