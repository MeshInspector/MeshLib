# message catalog compilation
find_program(MSGFMT_EXECUTABLE
  NAMES msgfmt
  PATHS ${GETTEXT_ROOT}
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

  add_custom_target(meshlib_translations
    DEPENDS ${MO_OUTPUT_FILES}
  )

  install(
    DIRECTORY ${ASSETS_OUTPUT_DIR}/locale
    DESTINATION ${MR_RESOURCES_DIR}
  )
endif(MSGFMT_EXECUTABLE)

# extraction of translatable strings
find_program(XGETTEXT_EXECUTABLE
  NAMES xgettext
  PATHS ${GETTEXT_ROOT}
  PATH_SUFFIXES bin
)
if(XGETTEXT_EXECUTABLE)
  file(REAL_PATH "${CMAKE_CURRENT_LIST_DIR}/.." SOURCE_ROOT_DIR)

  set(SOURCES "")
  foreach(LIB MRMesh MRIOExtras MRSymbolMesh MRVoxels MRViewer MRCommonPlugins)
    if(TARGET ${LIB})
      get_target_property(LIB_SOURCES ${LIB} SOURCES)
      foreach(SOURCE_FILE ${LIB_SOURCES})
        file(RELATIVE_PATH SOURCE_FILE ${SOURCE_ROOT_DIR} ${SOURCE_FILE})
        list(APPEND SOURCES ${SOURCE_FILE})
      endforeach()
    endif()
  endforeach()
  # use temporary file to bypass Windows' command line length limitations
  set(SOURCES_LIST_FILE "${CMAKE_CURRENT_BINARY_DIR}/xgettext_sources.list")
  list(JOIN SOURCES "\n" SOURCES_LIST)
  file(WRITE ${SOURCES_LIST_FILE} ${SOURCES_LIST})

  add_custom_target(meshlib_extract_strings
    COMMAND
      ${XGETTEXT_EXECUTABLE}
      -ktranslate:1,1t
      -ktranslate:1c,2,2t
      -ktranslate:1,2,3t
      -ktranslate:1c,2,3,4t
      -k_t:1,1t
      -k_t:1c,2,2t
      -k_t:1,2,3t
      -k_t:1c,2,3,4t
      -k_tr:1,1t
      -k_tr:1c,2,2t
      -k_tr:1,2,3t
      -k_tr:1c,2,3,4t
      -kf_tr:1,1t
      -kf_tr:1c,2,2t
      -kf_tr:1,2,3t
      -kf_tr:1c,2,3,4t
      --language=C++
      --from-code=UTF-8
      --add-comments="TRANSLATORS:"
      --default-domain=${MESHLIB_PROJECT_NAME}
      --output=${CMAKE_CURRENT_LIST_DIR}/${MESHLIB_PROJECT_NAME}.pot
      --files-from=${SOURCES_LIST_FILE}
    WORKING_DIRECTORY ${SOURCE_ROOT_DIR}
    DEPENDS ${SOURCES_LIST_FILE}
  )
endif()
