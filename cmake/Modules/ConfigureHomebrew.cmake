IF(APPLE)
  message("building for Apple")
  execute_process(
    COMMAND brew --prefix
    RESULT_VARIABLE CMD_ERROR
    OUTPUT_VARIABLE HOMEBREW_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  IF(CMD_ERROR EQUAL 0 AND EXISTS "${HOMEBREW_PREFIX}")
    message("Homebrew found. Prefix: ${HOMEBREW_PREFIX}")
  ELSE()
    message("Homebrew not found!")
    message(FATAL_ERROR "${CMD_ERROR} ${HOMEBREW_PREFIX}")
  ENDIF()

  include_directories(${HOMEBREW_PREFIX}/include)
  link_directories(${HOMEBREW_PREFIX}/lib)

  # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
  # TODO: revise
  set(CPPFLAGS "-I${HOMEBREW_PREFIX}/opt/llvm/include -I${HOMEBREW_PREFIX}/include")
  set(LDFLAGS "-L${HOMEBREW_PREFIX}/opt/llvm/lib -Wl,-rpath,${HOMEBREW_PREFIX}/opt/llvm/lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -undefined dynamic_lookup -framework Cocoa -framework OpenGL -framework IOKit") # https://github.com/pybind/pybind11/issues/382

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(MESHLIB_MACOS_FORCE_SYSTEM_LIBCXX)
      # use Homebrew zlib instead of system one for Clang builds
      execute_process(
        COMMAND brew --prefix zlib
        OUTPUT_VARIABLE HOMEBREW_ZLIB_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      set(ZLIB_ROOT ${HOMEBREW_ZLIB_PREFIX})
    else()
      # force use Clang's libc++
      if(CMAKE_CXX_COMPILER MATCHES "${HOMEBREW_PREFIX}")
        cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH LLVM_PREFIX)
        cmake_path(GET LLVM_PREFIX PARENT_PATH LLVM_PREFIX)
        add_compile_options(-nostdinc++ -isystem ${LLVM_PREFIX}/include/c++/v1)
        add_link_options(-nostdlib++ -L${LLVM_PREFIX}/lib -L${LLVM_PREFIX}/lib/c++ -Wl,-rpath,${LLVM_PREFIX}/lib -Wl,-rpath,${LLVM_PREFIX}/lib/c++ -lc++ -lc++abi)
      endif()
    endif()
  endif()
ENDIF() # APPLE
