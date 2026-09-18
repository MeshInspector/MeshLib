# emsdk 4.0.19 set this FALSE in its toolchain, so every add_library(... SHARED) came out
# static; 6.0.9 honours SHARED and builds side modules instead, whose link wants every input
# PIC - including the ports emsdk ships. Nothing in this build wants a side module.
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
