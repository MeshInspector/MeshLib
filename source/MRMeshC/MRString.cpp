#include "MRString.h"

#include "detail/TypeCast.h"

#include <string>

REGISTER_AUTO_CAST2( std::string, MRString )

const char* mrStringData( const MRString* str_ )
{
    ARG( str );
    return str.data();
}

size_t mrStringSize( const MRString* str_ )
{
    ARG( str );
    return str.size();
}

void mrStringFree( MRString* str_ )
{
    ARG_PTR( str );
    delete str;
}
