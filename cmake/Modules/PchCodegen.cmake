# `-fpch-codegen` support for the shared MRPch precompiled header (Clang/AppleClang only).
#
# `-fpch-codegen` homes code generated from PCH-instantiated templates and inline functions ONCE into a
# separate "PCH object" instead of letting every consuming translation unit emit (and the linker COMDAT-merge)
# its own copy. The win is the same one PR #6253 measured for the bindings PCH; here it is applied to the main
# build's PCH that is REUSE_FROM'd across ~16 targets / 642 TUs.
#
# The catch: with `-fpch-codegen` the homed symbols live only in the PCH object. They MUST be linked in or they
# come up undefined at link time (this is what historically made `-fpch-codegen` look "buggy"). MeshLib builds
# with `-fvisibility=hidden -fvisibility-inlines-hidden`, so those symbols are hidden per shared library and are
# NOT re-exported from one library to another. Therefore the single PCH object has to be linked into EVERY
# PCH-consuming target; each shared library / executable then carries its own private, hidden copy (the
# intra-library deduplication we are after). On non-Apple Clang `-Wl,-z,defs` turns a missed attachment into a
# hard link error, so an incomplete rollout fails loudly rather than at load time.
#
# CMake's `target_precompile_headers` has no built-in support for either `-fpch-codegen` or linking the PCH
# object, so the two functions below add that layer, mirroring scripts/mrbind/generate.mk.

# Compile MRPch's CMake-generated reused PCH into a single object file and wrap it in the `MRPchObject` target.
# Must be called from within source/MRPch/CMakeLists.txt (after target_precompile_headers), so that
# CMAKE_CURRENT_BINARY_DIR is MRPch's own binary dir and CMAKE_CXX_FLAGS already carries the codegen flags.
function(mr_build_pch_codegen_object)
  # CMake names the reused PCH for a single CXX precompiled header `cmake_pch.hxx.pch` (Clang/AppleClang).
  set(_pch "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MRPch.dir/cmake_pch.hxx.pch")
  set(_obj "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MRPch.dir/cmake_pch.hxx.pch.o")

  # Recompiling an already-built .pch into an object needs the codegen-affecting flags (visibility, -fPIC,
  # optimization / debug level) but NOT the PCH-producing flags themselves, exactly like generate.mk compiles
  # its `$1__PchObject` with the general compiler flags rather than the `-fpch-*` set. Strip the three codegen
  # flags so clang doesn't re-interpret them against a .pch input.
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _cfg)
  set(_obj_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_cfg}}")
  string(REPLACE "-fpch-codegen" "" _obj_flags "${_obj_flags}")
  string(REPLACE "-fpch-instantiate-templates" "" _obj_flags "${_obj_flags}")
  string(REPLACE "-fpch-debuginfo" "" _obj_flags "${_obj_flags}")
  separate_arguments(_obj_flags NATIVE_COMMAND "${_obj_flags}")

  set_source_files_properties("${_pch}" PROPERTIES GENERATED TRUE)
  add_custom_command(
    OUTPUT "${_obj}"
    # `-Wno-unused-command-line-argument`: a .pch input ignores most flags (include dirs, sysroot, std, ...);
    # without this, -Werror would turn each "argument unused during compilation" into a fatal error.
    COMMAND "${CMAKE_CXX_COMPILER}" ${_obj_flags} -Wno-unused-command-line-argument -c -o "${_obj}" "${_pch}"
    DEPENDS MRPch "${_pch}"
    VERBATIM
    COMMENT "Compiling shared PCH codegen object cmake_pch.hxx.pch.o"
  )
  add_custom_target(MRPchObject DEPENDS "${_obj}")

  # Publish the object path so mr_attach_pch_object() can reach it from other directories.
  set_property(GLOBAL PROPERTY MR_PCH_CODEGEN_OBJECT "${_obj}")
endfunction()

# Link the shared PCH object into a PCH-consuming target. Call from the consumer's own CMakeLists.txt, right
# after its `REUSE_FROM MRPch`, so set_source_files_properties() takes effect in the same directory scope as the
# target_sources() that uses it.
function(mr_attach_pch_object _target)
  get_property(_obj GLOBAL PROPERTY MR_PCH_CODEGEN_OBJECT)
  if(NOT _obj)
    message(FATAL_ERROR
      "mr_attach_pch_object(${_target}): PCH codegen object not registered; "
      "mr_build_pch_codegen_object() must run first (it does, from source/MRPch/CMakeLists.txt).")
  endif()
  # EXTERNAL_OBJECT: hand the prebuilt .o straight to the linker. GENERATED: it doesn't exist at configure time.
  set_source_files_properties("${_obj}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  target_sources(${_target} PRIVATE "${_obj}")
  add_dependencies(${_target} MRPchObject)
endfunction()
