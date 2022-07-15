#pragma once

#define STR(...) #__VA_ARGS__

#if defined( __clang__ )
#define SUPPRESS_WARNING_PUSH( x, y ) \
_Pragma( "clang diagnostic push" )    \
_Pragma( STR(clang diagnostic ignored x) )
#elif defined( __GNUC__ )
#define SUPPRESS_WARNING_PUSH( x, y ) \
_Pragma( "GCC diagnostic push ")       \
_Pragma( STR(GCC diagnostic ignored x) )
#elif defined( _MSC_VER )
#define SUPPRESS_WARNING_PUSH( x, y ) \
__pragma( warning( push ) )           \
__pragma( warning( disable: y ) )
#else
#define SUPPRESS_WARNING_PUSH( x, y )
#endif

#if defined( __clang__ )
#define SUPPRESS_WARNING_POP \
_Pragma( "clang diagnostic pop" )
#elif defined( __GNUC__ )
#define SUPPRESS_WARNING_POP \
_Pragma( "GCC diagnostic pop" )
#elif defined( _MSC_VER )
#define SUPPRESS_WARNING_POP \
__pragma( warning( pop ) )
#else
#define SUPPRESS_WARNING_POP
#endif
