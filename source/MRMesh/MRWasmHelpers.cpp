#ifdef __EMSCRIPTEN__
#include "MRPch/MRWasm.h"

extern "C"
{

/// Returns the size of a pointer on the current platform.
EMSCRIPTEN_KEEPALIVE int emsGetPointerSize()
{
    return sizeof (void*);
}

} // extern "C"
#endif
