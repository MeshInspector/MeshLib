if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  return()
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

function(opencascade_minimal_merged_library)
  separate_arguments(direct_toolkits UNIX_COMMAND "${OPENCASCADE_MINIMAL_DIRECT_TOOLKITS}")
  set(indirect_toolkits ${OCCT_LIBRARIES})
  list(REMOVE_ITEM indirect_toolkits ${direct_toolkits})

  file(WRITE "${CMAKE_BINARY_DIR}/OpenCASCADE.cxx" "")
  add_library(OpenCASCADE SHARED "${CMAKE_BINARY_DIR}/OpenCASCADE.cxx")
  target_link_libraries(OpenCASCADE PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${direct_toolkits}>" ${indirect_toolkits})
  target_include_directories(OpenCASCADE INTERFACE "$<INSTALL_INTERFACE:include/opencascade>" "$<INSTALL_INTERFACE:include>")
  if(APPLE)
    target_link_options(OpenCASCADE PRIVATE LINKER:-dead_strip)
  endif()

  install(TARGETS OpenCASCADE EXPORT opencascade-minimal-targets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
  install(EXPORT opencascade-minimal-targets NAMESPACE OpenCASCADE:: DESTINATION "${INSTALL_DIR_CMAKE}")
endfunction()

cmake_language(DEFER CALL opencascade_minimal_merged_library)
