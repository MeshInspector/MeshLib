# The icon folders of every resource directory are packed into one resource/icons.zip in the output directory,
# which MR::RibbonIcons reads on start instead of a thousand small files; the other resources are copied as files.
set(MR_ICON_FOLDERS_REGEX "/(icons|object_icons|independent_icons|logos)$")

# copies a resource directory without its icon folders and registers them for mr_add_icons_zip
function(mr_copy_resources SRC_DIR DST_DIR)
  get_filename_component(SRC_DIR "${SRC_DIR}" ABSOLUTE)
  file(COPY "${SRC_DIR}" DESTINATION "${DST_DIR}" REGEX "${MR_ICON_FOLDERS_REGEX}" EXCLUDE)
  set_property(GLOBAL APPEND PROPERTY MR_ICON_RESOURCE_DIRS "${SRC_DIR}")
endfunction()

# installs a resource directory without its icon folders
function(mr_install_resources SRC_DIR DST_DIR)
  install(DIRECTORY "${SRC_DIR}" DESTINATION "${DST_DIR}" REGEX "${MR_ICON_FOLDERS_REGEX}" EXCLUDE)
endfunction()

# packs the icons of all directories registered by mr_copy_resources into resource/icons.zip of the output directory;
# call it once from the top-level project after all add_subdirectory, with the application targets that must wait for the zip
function(mr_add_icons_zip)
  get_property(RESOURCE_DIRS GLOBAL PROPERTY MR_ICON_RESOURCE_DIRS)
  if(NOT RESOURCE_DIRS)
    return()
  endif()

  if(EMSCRIPTEN)
    set(ZIP_FILE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets/resource/icons.zip")
  else()
    set(ZIP_FILE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resource/icons.zip")
  endif()

  set(ICON_FILES "")
  foreach(RESOURCE_DIR ${RESOURCE_DIRS})
    file(GLOB_RECURSE DIR_ICON_FILES CONFIGURE_DEPENDS
      "${RESOURCE_DIR}/icons/*.png"
      "${RESOURCE_DIR}/object_icons/*.png"
      "${RESOURCE_DIR}/independent_icons/*.png"
      "${RESOURCE_DIR}/logos/*.png"
    )
    list(APPEND ICON_FILES ${DIR_ICON_FILES})
  endforeach()

  if(NOT Python_EXECUTABLE)
    find_package(Python REQUIRED COMPONENTS Interpreter)
  endif()
  set(PACK_SCRIPT "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../scripts/pack_icons.py")
  add_custom_command(
    OUTPUT "${ZIP_FILE}"
    COMMAND ${Python_EXECUTABLE} "${PACK_SCRIPT}" "${ZIP_FILE}" ${RESOURCE_DIRS}
    DEPENDS ${ICON_FILES} "${PACK_SCRIPT}"
    COMMENT "Packing icons into ${ZIP_FILE}"
  )
  add_custom_target(MRIconsZip ALL DEPENDS "${ZIP_FILE}")
  foreach(APP_TARGET ${ARGN})
    add_dependencies(${APP_TARGET} MRIconsZip)
  endforeach()
  install(FILES "${ZIP_FILE}" DESTINATION "${MR_RESOURCES_DIR}/resource")
endfunction()
