# Precompiled-header configuration.
# Included from the top-level CMakeLists right before CompilerOptions, so it must come AFTER the
# `project' command (it relies on the detected compiler information).

set(MR_PCH_DEFAULT ON)
# for Clang<15 builds: PCH not only does not give any speedup, but even vice versa
IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15)
  set(MR_PCH_DEFAULT OFF)
ENDIF()
set(MR_PCH ${MR_PCH_DEFAULT} CACHE BOOL "Enable precompiled headers")
IF(MR_PCH AND NOT MR_EMSCRIPTEN AND NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
ENDIF()
message("MR_PCH=${MR_PCH}")

# On MSVC the *_API macros expand to __declspec(dllexport) inside a library and to
# __declspec(dllimport) in its consumers, so MeshLib's own headers cannot be baked into
# a single shared precompiled header. Instead we enable the extra headers everywhere and
# let MRMesh and MRViewer build their own PCH from MRPch.h in their (dllexport) context,
# while every other target keeps reusing the shared (all-dllimport) PCH built by MRPch.
# See source/MRMesh/CMakeLists.txt and source/MRViewer/CMakeLists.txt.
IF(MSVC AND MR_PCH)
  set(MR_PCH_USE_EXTRA_HEADERS ON CACHE BOOL "Add frequently used MeshLib headers to the precompiled header" FORCE)
  add_compile_definitions(MR_PCH_USE_EXTRA_HEADERS)
ENDIF()

# Caching OpenVDB in the precompiled header speeds up the OpenVDB-heavy translation units, but it only
# works on MSVC: there MRMesh/MRViewer build their own PCH and the compiler does not emit unreferenced
# inline functions, so the many targets that REUSE_FROM the shared PCH without linking libopenvdb still
# link. On GCC/Clang that emission yields undefined OpenVDB symbols (e.g. openvdb::math::simplify from
# inline ScaleMap methods), so the option has no effect there. It also noticeably enlarges the PCH.
option(MR_PCH_OPENVDB "Cache OpenVDB in the precompiled header (effective on MSVC only)" ON)

# MRPch.h includes OpenVDB unless MRPCH_NO_OPENVDB is defined. The macro affects the shared PCH, so it
# must be defined identically on the PCH and on every REUSE_FROM consumer (otherwise GCC errors with
# -Werror=invalid-pch). Define it globally whenever OpenVDB must stay out of the PCH: always on non-MSVC,
# and on MSVC when the option is disabled.
IF(MR_PCH AND (NOT MSVC OR NOT MR_PCH_OPENVDB))
  add_compile_definitions(MRPCH_NO_OPENVDB)
ENDIF()
