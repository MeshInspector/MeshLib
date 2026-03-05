# message catalog compilation
find_program(MSGFMT_EXECUTABLE
  NAMES msgfmt
  PATHS ${GETTEXT_ROOT} $ENV{GETTEXT_ROOT}
  PATH_SUFFIXES bin
)
if(MSGFMT_EXECUTABLE)
  if(EMSCRIPTEN)
    set(ASSETS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
  else()
    set(ASSETS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  endif()

  set(MO_OUTPUT_FILES "")
  file(GLOB_RECURSE PO_FILES "*.po")
  foreach(PO_FILE ${PO_FILES})
    get_filename_component(LOCALE_DIR ${PO_FILE} DIRECTORY)
    get_filename_component(LOCALE_NAME ${LOCALE_DIR} NAME)
    get_filename_component(DOMAIN_NAME ${PO_FILE} NAME_WLE)
    message(STATUS "Found ${LOCALE_NAME} locale for ${DOMAIN_NAME}")

    set(MO_OUTPUT_DIR "${ASSETS_OUTPUT_DIR}/locale/${LOCALE_NAME}/LC_MESSAGES")
    set(MO_OUTPUT_FILE "${MO_OUTPUT_DIR}/${DOMAIN_NAME}.mo")
    list(APPEND MO_OUTPUT_FILES ${MO_OUTPUT_FILE})

    add_custom_command(
      OUTPUT ${MO_OUTPUT_FILE}
      MAIN_DEPENDENCY ${PO_FILE}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${MO_OUTPUT_DIR}"
      COMMAND ${MSGFMT_EXECUTABLE} ${PO_FILE} --output-file=${MO_OUTPUT_FILE} --check
    )
  endforeach(PO_FILE)

  if(MO_OUTPUT_FILES)
    add_custom_target(meshlib_translations
      ALL
      DEPENDS ${MO_OUTPUT_FILES}
    )

    install(
      DIRECTORY ${ASSETS_OUTPUT_DIR}/locale
      DESTINATION ${MR_RESOURCES_DIR}
    )
  endif(MO_OUTPUT_FILES)
else(MSGFMT_EXECUTABLE)
  message(WARNING "No msgfmt executable was found, unable to compile translation files.")
endif(MSGFMT_EXECUTABLE)
