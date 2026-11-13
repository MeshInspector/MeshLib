function(mr_add_wayland_protocols TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs PROTOCOL_PATHS)
  cmake_parse_arguments(WAYLAND "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  find_package(PkgConfig REQUIRED)

  pkg_check_modules(wayland-client REQUIRED IMPORTED_TARGET wayland-client)
  target_link_libraries(${TARGET_NAME} PRIVATE PkgConfig::wayland-client)

  pkg_check_modules(wayland-protocols REQUIRED IMPORTED_TARGET wayland-protocols)
  pkg_get_variable(WAYLAND_PROTOCOLS_DATADIR wayland-protocols pkgdatadir)

  find_program(WAYLAND_SCANNER wayland-scanner REQUIRED)

  set(WAYLAND_PROTOCOLS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/wayland-output")
  file(MAKE_DIRECTORY "${WAYLAND_PROTOCOLS_OUTPUT_DIR}")
  target_include_directories(${PROJECT_NAME} PRIVATE "${WAYLAND_PROTOCOLS_OUTPUT_DIR}")

  foreach(PROTOCOL_PATH ${WAYLAND_PROTOCOL_PATHS})
    get_filename_component(PROTOCOL_NAME "${PROTOCOL_PATH}" NAME)
    set(PROTOCOL_XML "${WAYLAND_PROTOCOLS_DATADIR}/${PROTOCOL_PATH}.xml")
    set(PROTOCOL_CLIENT_HEADER "${WAYLAND_PROTOCOLS_OUTPUT_DIR}/${PROTOCOL_NAME}-client-protocol.h")
    set(PROTOCOL_PRIVATE_CODE "${WAYLAND_PROTOCOLS_OUTPUT_DIR}/${PROTOCOL_NAME}-protocol.c")

    add_custom_command(
      OUTPUT "${PROTOCOL_CLIENT_HEADER}"
      COMMAND ${WAYLAND_SCANNER} client-header "${PROTOCOL_XML}" "${PROTOCOL_CLIENT_HEADER}"
      DEPENDS "${PROTOCOL_XML}"
      VERBATIM
    )
    add_custom_command(
      OUTPUT "${PROTOCOL_PRIVATE_CODE}"
      COMMAND ${WAYLAND_SCANNER} private-code "${PROTOCOL_XML}" "${PROTOCOL_PRIVATE_CODE}"
      DEPENDS "${PROTOCOL_XML}"
      VERBATIM
    )
    target_sources(${TARGET_NAME}
      PRIVATE
      "${PROTOCOL_CLIENT_HEADER}"
      "${PROTOCOL_PRIVATE_CODE}"
    )
  endforeach()
endfunction()