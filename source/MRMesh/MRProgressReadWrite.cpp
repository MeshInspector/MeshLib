#include "MRProgressReadWrite.h"

namespace MR
{

bool writeByBlocks( std::ostream& out, const char* data, size_t dataSize, ProgressCallback callback /*= {}*/, size_t blockSize /*= ( size_t( 1 ) << 16 )*/ )
{
    if ( !dataSize )
        return true;

    if ( !callback )
    {
        out.write( data, dataSize );
        return true;
    }

    int blockIndex = 0;
    for ( size_t max = dataSize / blockSize; blockIndex < max; ++blockIndex )
    {
        out.write( data + blockIndex * blockSize, blockSize );
        if ( !callback( float( blockIndex * blockSize ) / dataSize ) )
            return false;
    }
    const size_t remnant = dataSize - blockIndex * blockSize;
    if ( remnant )
        out.write( data + blockIndex * blockSize, remnant );
    if ( !callback( float( blockIndex * blockSize + remnant ) / dataSize ) )
        return false;

    return true;
}

bool readByBlocks( std::istream& in, char* data, size_t dataSize, ProgressCallback callback /*= {}*/, size_t blockSize /*= ( size_t( 1 ) << 16 )*/ )
{
    if ( !dataSize )
        return true;

    if ( !callback )
    {
        in.read( data, dataSize );
        return true;
    }

    int blockIndex = 0;
    for ( size_t max = dataSize / blockSize; blockIndex < max; ++blockIndex )
    {
        in.read( data + blockIndex * blockSize, blockSize );
        if ( !callback( float( blockIndex * blockSize ) / dataSize ) )
            return false;
    }
    const size_t remnant = dataSize - blockIndex * blockSize;
    if ( remnant )
        in.read( data + blockIndex * blockSize, remnant );
    if ( !callback( float( blockIndex * blockSize + remnant ) / dataSize ) )
        return false;

    return true;
}

}
