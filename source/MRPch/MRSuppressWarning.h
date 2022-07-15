#pragma once

#define MR_STR(...) #__VA_ARGS__

#if defined( __clang__ )
#define MR_SUPPRESS_WARNING_PUSH( x, y ) \
_Pragma( "clang diagnostic push" )    \
_Pragma( MR_STR(clang diagnostic ignored x) )
#elif defined( __GNUC__ )
#define MR_SUPPRESS_WARNING_PUSH( x, y ) \
_Pragma( "GCC diagnostic push ")       \
_Pragma( MR_STR(GCC diagnostic ignored x) )
#elif defined( _MSC_VER )
#define MR_SUPPRESS_WARNING_PUSH( x, y ) \
__pragma( warning( push ) )           \
__pragma( warning( disable: y ) )
#else
#define MR_SUPPRESS_WARNING_PUSH( x, y )
#endif

#if defined( __clang__ )
#define MR_SUPPRESS_WARNING_POP \
_Pragma( "clang diagnostic pop" )
#elif defined( __GNUC__ )
#define MR_SUPPRESS_WARNING_POP \
_Pragma( "GCC diagnostic pop" )
#elif defined( _MSC_VER )
#define MR_SUPPRESS_WARNING_POP \
__pragma( warning( pop ) )
#else
#define MR_SUPPRESS_WARNING_POP
#endif
