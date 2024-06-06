set(MR_PLATFORM "UNKNOWN")
IF(MR_EMSCRIPTEN)
  set(MR_PLATFORM "WASM")
ELSEIF(APPLE)
  set(MR_PLATFORM "APPLE_${CMAKE_SYSTEM_PROCESSOR}")
ELSEIF(WIN32)
  set(MR_PLATFORM "Windows")
ELSEIF(EXISTS /etc/os-release)
  # TODO: use ID variable
  file(STRINGS /etc/os-release distro REGEX "^NAME=")
  string(REGEX REPLACE "NAME=\"(.*)\"" "\\1" distro "${distro}")

  file(STRINGS /etc/os-release version_id REGEX "^VERSION_ID=")
  string(REGEX REPLACE "VERSION_ID=(.*)" "\\1" version_id "${version_id}")
  string(REGEX REPLACE "\"(.*)\"" "\\1" version_id "${version_id}")

  # TODO: don't strip Ubuntu version (use 24.04 instead of 24)
  IF(${distro} STREQUAL "Ubuntu")
    string(FIND ${version_id} "." dot-pos)
    string(SUBSTRING ${version_id} 0 ${dot-pos} version_id)
  ENDIF()

  set(MR_PLATFORM "${distro}_${version_id}")
  set(MR_LINUX_DISTRO "${distro}")
ENDIF()

add_compile_definitions(MR_PLATFORM="${MR_PLATFORM}")
message("platform: ${MR_PLATFORM}")
