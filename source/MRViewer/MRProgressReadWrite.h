#pragma once
#include "MRMesh/MRProgressCallback.h"
#include <ostream>

namespace MR
{

inline bool writeWithProgress( std::ostream& out, const char* data, size_t size, ProgressCallback callback = {}, size_t begin = 0, size_t sizeAll = 0 )
{
    if ( !callback )
    {
        out.write( data, size );
        return true;
    }

    if ( !sizeAll )
        sizeAll = size;

    const size_t blockSize = size_t( 1 ) << 16;
    int blockIndex = 0;
    for ( size_t max = size / blockSize; blockIndex < max; ++blockIndex )
    {
        out.write( data + blockIndex * blockSize, blockSize );
        if ( callback && !callback( float( begin + blockIndex * blockSize ) / sizeAll ) )
            return false;
    }
    const size_t remnant = size - blockIndex * blockSize;
    if ( remnant )
        out.write( data + blockIndex * blockSize, remnant );
    if ( callback && !callback( float( begin + blockIndex * blockSize + remnant ) / sizeAll ) )
        return false;

    return true;
}

}
