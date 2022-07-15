#pragma once

// Helper macro for concatenating arguments into string literal.
#define MR_STR(...) #__VA_ARGS__

// Macros for disabling compiler warnings.
// `MESSAGE' is diagnostic message name used by Clang and GCC.
// `NUMBER` is warning number used by MSVC.
#if defined( __clang__ )
#define MR_SUPPRESS_WARNING_PUSH( MESSAGE, NUMBER ) \
_Pragma( "clang diagnostic push" )                  \
_Pragma( MR_STR(clang diagnostic ignored MESSAGE) )
#define MR_SUPPRESS_WARNING_POP \
_Pragma( "clang diagnostic pop" )
#elif defined( __GNUC__ )
#define MR_SUPPRESS_WARNING_PUSH( MESSAGE, NUMBER ) \
_Pragma( "GCC diagnostic push ")                    \
_Pragma( MR_STR(GCC diagnostic ignored MESSAGE) )
#define MR_SUPPRESS_WARNING_POP \
_Pragma( "GCC diagnostic pop" )
#elif defined( _MSC_VER )
#define MR_SUPPRESS_WARNING_PUSH( MESSAGE, NUMBER ) \
__pragma( warning( push ) )                         \
__pragma( warning( disable: NUMBER ) )
#define MR_SUPPRESS_WARNING_POP \
__pragma( warning( pop ) )
#else
#define MR_SUPPRESS_WARNING_PUSH( MESSAGE, NUMBER )
#define MR_SUPPRESS_WARNING_POP
#endif
