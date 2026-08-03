IF(APPLE)
  message("building for Apple")
  # Allow an explicit Homebrew prefix override (e.g. -D HOMEBREW_PREFIX=/usr/local
  # to link the x86_64 bottles when cross-building Intel on an arm64 host).
  # Falls back to `brew --prefix` for the common native case.
  IF(NOT HOMEBREW_PREFIX)
    execute_process(
      COMMAND brew --prefix
      OUTPUT_VARIABLE HOMEBREW_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  ENDIF()
  # Validate whichever prefix we ended up with (auto-detected or overridden).
  IF(NOT EXISTS "${HOMEBREW_PREFIX}")
    message(FATAL_ERROR "Homebrew prefix not found: '${HOMEBREW_PREFIX}'")
  ENDIF()
  message("Homebrew prefix: ${HOMEBREW_PREFIX}")

  include_directories(${HOMEBREW_PREFIX}/include)
  link_directories(${HOMEBREW_PREFIX}/lib)

  # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
  # TODO: revise
  set(CPPFLAGS "-I${HOMEBREW_PREFIX}/opt/llvm/include -I${HOMEBREW_PREFIX}/include")
  set(LDFLAGS "-L${HOMEBREW_PREFIX}/opt/llvm/lib -Wl,-rpath,${HOMEBREW_PREFIX}/opt/llvm/lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -undefined dynamic_lookup -framework Cocoa -framework OpenGL -framework IOKit") # https://github.com/pybind/pybind11/issues/382

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # use Homebrew zlib instead of system one for Clang builds
    execute_process(
      COMMAND brew --prefix zlib
      OUTPUT_VARIABLE HOMEBREW_ZLIB_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(ZLIB_ROOT ${HOMEBREW_ZLIB_PREFIX})
  endif()
ENDIF() # APPLE
