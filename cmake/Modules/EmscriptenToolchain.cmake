# emsdk 4.0.19 set this in its own toolchain file, so every add_library(... SHARED) came out
# static; 6.0.9 dropped it and builds real side modules instead, whose link wants every input
# PIC - including the ports emsdk ships. Nothing here wants a side module.
# It has to be set before project(), hence a toolchain wrapper rather than an include.
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)

include("$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake")
