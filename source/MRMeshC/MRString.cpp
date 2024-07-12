#include "MRString.h"

#include <string>

const char* mrStringData( const MRString* str )
{
    return reinterpret_cast<const std::string*>( str )->data();
}

size_t mrStringSize( const MRString* str )
{
    return reinterpret_cast<const std::string*>( str )->size();
}

void mrStringFree( MRString* str )
{
    delete reinterpret_cast<std::string*>( str );
}
