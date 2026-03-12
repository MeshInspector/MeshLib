find_program(MSGFMT_EXECUTABLE
  NAMES msgfmt
  PATHS ${GETTEXT_ROOT} $ENV{GETTEXT_ROOT}
  PATH_SUFFIXES bin
)

function(mr_add_translations TARGET_NAME)
  if(MSGFMT_EXECUTABLE)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs DOMAINS PATHS)
    cmake_parse_arguments(LOCALE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(EMSCRIPTEN)
      set(ASSETS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")
    else()
      set(ASSETS_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    endif()

    set(POT_FILES "")
    set(MO_OUTPUT_FILES "")
    foreach(LOCALE_DIR ${LOCALE_PATHS})
      file(GLOB LOCALE_NAMES LIST_DIRECTORIES true RELATIVE "${LOCALE_DIR}" "${LOCALE_DIR}/*")

      foreach(DOMAIN_NAME ${LOCALE_DOMAINS})
        set(POT_FILE "${LOCALE_DIR}/${DOMAIN_NAME}.pot")
        if(NOT EXISTS ${POT_FILE})
          continue()
        endif()
        list(APPEND POT_FILES ${POT_FILE})

        set(FOUND_LOCALES "")
        foreach(LOCALE_NAME ${LOCALE_NAMES})
          set(PO_FILE "${LOCALE_DIR}/${LOCALE_NAME}/${DOMAIN_NAME}.po")
          if(NOT EXISTS ${PO_FILE})
            continue()
          endif()
          list(APPEND FOUND_LOCALES ${LOCALE_NAME})

          set(MO_OUTPUT_DIR "${ASSETS_OUTPUT_DIR}/locale/${LOCALE_NAME}/LC_MESSAGES")
          set(MO_OUTPUT_FILE "${MO_OUTPUT_DIR}/${DOMAIN_NAME}.mo")
          list(APPEND MO_OUTPUT_FILES ${MO_OUTPUT_FILE})

          add_custom_command(
            OUTPUT ${MO_OUTPUT_FILE}
            MAIN_DEPENDENCY ${PO_FILE}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MO_OUTPUT_DIR}"
            COMMAND ${MSGFMT_EXECUTABLE} ${PO_FILE} --output-file=${MO_OUTPUT_FILE} --check
          )
        endforeach(LOCALE_NAME)

        list(LENGTH FOUND_LOCALES FOUND_LOCALES_COUNT)
        if(${FOUND_LOCALES_COUNT} EQUAL 1)
          message(STATUS "Found ${FOUND_LOCALES_COUNT} translation for ${DOMAIN_NAME}.")
        else()
          message(STATUS "Found ${FOUND_LOCALES_COUNT} translations for ${DOMAIN_NAME}.")
        endif()
      endforeach(DOMAIN_NAME)
    endforeach(LOCALE_DIR)

    if(POT_FILES)
      install(
        FILES ${POT_FILES}
        DESTINATION ${MR_RESOURCES_DIR}/locale
      )
    endif(POT_FILES)

    if(MO_OUTPUT_FILES)
      add_custom_target(${TARGET_NAME}
        DEPENDS ${MO_OUTPUT_FILES}
      )

      install(
        DIRECTORY ${ASSETS_OUTPUT_DIR}/locale/
        DESTINATION ${MR_RESOURCES_DIR}/locale
      )
    endif(MO_OUTPUT_FILES)
  endif(MSGFMT_EXECUTABLE)
endfunction(mr_add_translations)
