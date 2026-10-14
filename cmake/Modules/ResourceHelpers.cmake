# The icon folders of a resource directory (icons, object_icons, independent_icons, logos) are packed into
# resource/icons-<target>.zip in the output directory, which MR::RibbonIcons reads on start instead of
# a thousand small files; the other resources are copied as files.
set(MR_ICON_FOLDERS_REGEX "/(icons|object_icons|independent_icons|logos)$")

# copies the resource directory of a target to the output directory and installs it,
# with the icon folders packed into resource/icons-<target>.zip
function(mr_add_resources TARGET_NAME)
  set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resource")
  if(EMSCRIPTEN)
    set(DST_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
  else()
    set(DST_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  endif()
  file(COPY "${SRC_DIR}" DESTINATION "${DST_DIR}" REGEX "${MR_ICON_FOLDERS_REGEX}" EXCLUDE)
  install(DIRECTORY "${SRC_DIR}" DESTINATION "${MR_RESOURCES_DIR}" REGEX "${MR_ICON_FOLDERS_REGEX}" EXCLUDE)

  file(GLOB_RECURSE ICON_FILES CONFIGURE_DEPENDS
    "${SRC_DIR}/icons/*.png"
    "${SRC_DIR}/object_icons/*.png"
    "${SRC_DIR}/independent_icons/*.png"
    "${SRC_DIR}/logos/*.png"
  )
  if(NOT ICON_FILES)
    return()
  endif()

  if(NOT Python_EXECUTABLE)
    find_package(Python REQUIRED COMPONENTS Interpreter)
  endif()
  set(PACK_SCRIPT "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../scripts/pack_icons.py")
  set(ZIP_FILE "${DST_DIR}/resource/icons-${TARGET_NAME}.zip")
  add_custom_command(
    OUTPUT "${ZIP_FILE}"
    COMMAND ${Python_EXECUTABLE} "${PACK_SCRIPT}" "${ZIP_FILE}" "${SRC_DIR}"
    DEPENDS ${ICON_FILES} "${PACK_SCRIPT}"
    COMMENT "Packing icons into ${ZIP_FILE}"
  )
  add_custom_target(${TARGET_NAME}Icons DEPENDS "${ZIP_FILE}")
  add_dependencies(${TARGET_NAME} ${TARGET_NAME}Icons)
  install(FILES "${ZIP_FILE}" DESTINATION "${MR_RESOURCES_DIR}/resource")
endfunction()
