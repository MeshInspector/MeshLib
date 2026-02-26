# message catalog compilation
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

# extraction of translatable strings
find_program(XGETTEXT_EXECUTABLE
  NAMES xgettext
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
      ${SOURCES}
    WORKING_DIRECTORY ${SOURCE_ROOT_DIR}
  )
endif()
