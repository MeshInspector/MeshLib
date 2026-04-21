include(CMakeFindDependencyMacro)

IF(NOT MESHLIB_USE_VCPKG AND NOT APPLE)
  find_dependency(fastmcpp)
endif()
