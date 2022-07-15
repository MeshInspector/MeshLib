#pragma once

#if defined( __clang__ )
#define SUPPRESS_DEPRECATION_WARNING( x ) \
_Pragma( "clang diagnostic push" )      \
_Pragma( "clang diagnostic ignored \"-Wdeprecated-declarations\"" ) \
{ x }                                   \
_Pragma( "clang diagnostic pop" )
#elif defined( __GNUC__ )
#define SUPPRESS_DEPRECATION_WARNING( x ) \
_Pragma( "GCC diagnostic push" )        \
_Pragma( "GCC diagnostic ignored \"-Wdeprecated-declarations\"" ) \
{ x }                                   \
_Pragma( "GCC diagnostic pop" )
#elif defined( _MSC_VER )
#define SUPPRESS_DEPRECATION_WARNING( x ) \
_pragma( warning( push ) )              \
_pragma( warning( disable: 4996 ) )     \
{ x }                                   \
_pragma( warning( pop ) )
#else
#define SUPPRESS_DEPRECATION_WARNING( x ) { x }
#endif
