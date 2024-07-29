#pragma once

#include "MRMesh/MRMacros.h"

// Macros for disabling compiler warnings.
// `MESSAGE' is diagnostic message name used by Clang and GCC.
// `NUMBER` is warning number used by MSVC.
#if defined( __clang__ )
    #define MR_SUPPRESS_WARNING_PUSH \
        _Pragma( "clang diagnostic push" )
    #define MR_SUPPRESS_WARNING( MESSAGE, NUMBER ) \
        _Pragma( MR_STR(clang diagnostic ignored MESSAGE) )
    #define MR_SUPPRESS_WARNING_POP \
        _Pragma( "clang diagnostic pop" )
#elif defined( __GNUC__ )
    #define MR_SUPPRESS_WARNING_PUSH \
        _Pragma( "GCC diagnostic push" )
    #define MR_SUPPRESS_WARNING( MESSAGE, NUMBER ) \
        _Pragma( MR_STR(GCC diagnostic ignored MESSAGE) )
    #define MR_SUPPRESS_WARNING_POP \
        _Pragma( "GCC diagnostic pop" )
#elif defined( _MSC_VER )
    #define MR_SUPPRESS_WARNING_PUSH \
        __pragma( warning( push ) )
    #define MR_SUPPRESS_WARNING( MESSAGE, NUMBER ) \
        __pragma( warning( disable: NUMBER ) )
    #define MR_SUPPRESS_WARNING_POP \
        __pragma( warning( pop ) )
#else
    #define MR_SUPPRESS_WARNING_PUSH
    #define MR_SUPPRESS_WARNING( MESSAGE, NUMBER )
    #define MR_SUPPRESS_WARNING_POP
#endif
