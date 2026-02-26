find_program(MSGFMT_EXECUTABLE
  NAMES msgfmt
)
if(MSGFMT_EXECUTABLE)
  add_custom_target(meshlib_translations ALL)

  file(GLOB_RECURSE PO_FILES "*.po")
  foreach(PO_FILE ${PO_FILES})
    get_filename_component(LOCALE_DIR ${PO_FILE} DIRECTORY)
    get_filename_component(LOCALE_NAME ${LOCALE_DIR} NAME)
    get_filename_component(DOMAIN_NAME ${PO_FILE} NAME_WLE)
    set(MO_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/locale/${LOCALE_NAME}/LC_MESSAGES")
    set(MO_OUTPUT_FILE "${MO_OUTPUT_DIR}/${DOMAIN_NAME}.mo")

    file(MAKE_DIRECTORY ${MO_OUTPUT_DIR})
    add_custom_target(meshlib_translation_${LOCALE_NAME}
      COMMAND ${MSGFMT_EXECUTABLE} ${PO_FILE} --output-file=${MO_OUTPUT_FILE} --check
      DEPENDS ${PO_FILE}
    )

    add_dependencies(meshlib_translations meshlib_translation_${LOCALE_NAME})
  endforeach(PO_FILE)
endif(MSGFMT_EXECUTABLE)
