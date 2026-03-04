find_program(MSGFMT_EXECUTABLE
  NAMES msgfmt
  PATHS ${GETTEXT_ROOT} $ENV{GETTEXT_ROOT}
  PATH_SUFFIXES bin
)

function(mr_add_translations TARGET_NAME POT_FILE)
  if(MSGFMT_EXECUTABLE)
    if(EMSCRIPTEN)
      set(ASSETS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
    else()
      set(ASSETS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    endif()

    get_filename_component(DOMAIN_NAME ${POT_FILE} NAME_WLE)
    get_filename_component(LOCALE_ROOT_DIR ${POT_FILE} DIRECTORY)
    get_filename_component(LOCALE_ROOT_DIR ${LOCALE_ROOT_DIR} ABSOLUTE)

    set(MO_OUTPUT_FILES "")
    file(GLOB LOCALE_NAMES LIST_DIRECTORIES true RELATIVE "${LOCALE_ROOT_DIR}" "${LOCALE_ROOT_DIR}/*")
    foreach(LOCALE_NAME ${LOCALE_NAMES})
      set(PO_FILE "${LOCALE_ROOT_DIR}/${LOCALE_NAME}/${DOMAIN_NAME}.po")
      if(EXISTS ${PO_FILE})
        set(MO_OUTPUT_DIR "${ASSETS_OUTPUT_DIR}/locale/${LOCALE_NAME}/LC_MESSAGES")
        set(MO_OUTPUT_FILE "${MO_OUTPUT_DIR}/${DOMAIN_NAME}.mo")
        list(APPEND MO_OUTPUT_FILES ${MO_OUTPUT_FILE})

        add_custom_command(
          OUTPUT ${MO_OUTPUT_FILE}
          MAIN_DEPENDENCY ${PO_FILE}
          COMMAND ${CMAKE_COMMAND} -E make_directory "${MO_OUTPUT_DIR}"
          COMMAND ${MSGFMT_EXECUTABLE} ${PO_FILE} --output-file=${MO_OUTPUT_FILE} --check
        )
      endif()
    endforeach(LOCALE_NAME)

    if(MO_OUTPUT_FILES)
      list(LENGTH ${MO_OUTPUT_FILES} MO_OUTPUT_FILES_COUNT)
      if(${MO_OUTPUT_FILES_COUNT} EQUAL 1)
        message(STATUS "Found ${MO_OUTPUT_FILES_COUNT} translation for ${DOMAIN_NAME}.")
      else()
        message(STATUS "Found ${MO_OUTPUT_FILES_COUNT} translations for ${DOMAIN_NAME}.")
      endif()

      add_custom_target(${TARGET_NAME}
        DEPENDS ${MO_OUTPUT_FILES}
      )

      install(
        FILES ${POT_FILE}
        DESTINATION ${MR_RESOURCES_DIR}/locale
      )
      install(
        DIRECTORY ${ASSETS_OUTPUT_DIR}/locale/
        DESTINATION ${MR_RESOURCES_DIR}/locale
        FILES_MATCHING
        PATTERN "${DOMAIN_NAME}.mo"
      )
    endif(MO_OUTPUT_FILES)
  endif(MSGFMT_EXECUTABLE)
endfunction()
