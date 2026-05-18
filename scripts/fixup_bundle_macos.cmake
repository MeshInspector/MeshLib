cmake_minimum_required(VERSION 3.18)

# Wrapper around CMake's BundleUtilities::fixup_bundle(), invoked via
# `cmake -DAPP=... -DLIBS=... -DDIRS=... -P scripts/fixup_bundle_macos.cmake`
# from scripts/distribution_apple.sh.
#
# MeshLib's framework is a non-standard layout (multiple executables in bin/,
# MeshLib's own dylibs in lib/, thirdparty-built libs in lib/lib/, Python .so
# modules under lib/lib/meshlib/), so we pass:
#   APP   absolute path to one in-bundle executable. fixup_bundle treats
#         its parent directory as the bundle dir and seeds the prereq walk
#         from there.
#   LIBS  ';'-separated list of every other Mach-O already shipped inside
#         the framework (binaries, dylibs, .so modules). They are passed
#         explicitly so fixup_bundle records them as in-bundle items --
#         it won't try to re-copy or re-resolve them. CMake's
#         get_bundle_all_executables() only picks up files whose `file -b`
#         output matches the word "executable", so Mach-O dylibs and
#         Mach-O bundles (Python .so) are otherwise invisible to it.
#   DIRS  ';'-separated list of directories searched when resolving any
#         non-absolute prerequisite (@rpath/... etc.). Caller assembles
#         this from the active Homebrew prefix discovered at runtime
#         (`brew --prefix`), so the self-hosted arm64 runner whose prefix
#         is /Users/runner/.homebrew works the same as the standard hosts.

include(BundleUtilities)

foreach(_var IN ITEMS APP LIBS DIRS)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR "Required variable ${_var} not set (pass via -D${_var}=...).")
  endif()
endforeach()

if(NOT EXISTS "${APP}")
  message(FATAL_ERROR "APP does not exist: ${APP}")
endif()

# Override default embedded location for bundled prerequisites. Without this
# fixup_bundle would copy them to <bundle>/Contents/Frameworks/, which doesn't
# exist in MeshLib.framework's layout. With this override they land in
# <framework>/lib/ (the same dir that already holds MeshLib's own dylibs),
# and load commands become @executable_path/../lib/<basename>.
function(gp_item_default_embedded_path_override item embedded_path_var)
  set(${embedded_path_var} "@executable_path/../lib" PARENT_SCOPE)
endfunction()

# Homebrew bottles install dylibs read-only; without this install_name_tool
# would fail on the first bundled lib it tries to edit.
set(BU_CHMOD_BUNDLE_ITEMS TRUE)

list(LENGTH LIBS _n_libs)
list(LENGTH DIRS _n_dirs)
message(STATUS "fixup_bundle APP:  ${APP}")
message(STATUS "fixup_bundle LIBS (${_n_libs} items):")
foreach(_lib IN LISTS LIBS)
  message(STATUS "  ${_lib}")
endforeach()
message(STATUS "fixup_bundle DIRS (${_n_dirs} items):")
foreach(_d IN LISTS DIRS)
  message(STATUS "  ${_d}")
endforeach()

# IGNORE_ITEM keeps the embedded Python framework and libpython3.10.dylib
# external -- the same exclusion the previous Python bundler had. Matched
# by basename.
fixup_bundle("${APP}" "${LIBS}" "${DIRS}"
  IGNORE_ITEM "Python" "libpython3.10.dylib")
