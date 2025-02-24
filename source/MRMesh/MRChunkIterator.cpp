#include "MRChunkIterator.h"

namespace MR
{

ChunkIterator& ChunkIterator::operator++()
{
    ++index;
    return *this;
}

Chunk ChunkIterator::operator*() const
{
    const auto offset = index * ( chunkSize - overlap );
    return {
        .offset = offset,
        .size = std::min( chunkSize, totalSize - offset ),
    };
}

size_t chunkCount( size_t totalSize, size_t chunkSize, size_t overlap )
{
    if ( totalSize == 0 || chunkSize == 0 || chunkSize <= overlap )
        return 0;

    const auto size = totalSize - overlap; // otherwise the last chunk's size may be smaller or equal to the overlap i.e. fully in the previous chunk
    const auto step = chunkSize - overlap;
    return ( size / step ) + !!( size % step ); // integer variant of `std::ceil( a / b )`
}

IteratorRange<ChunkIterator> splitByChunks( size_t totalSize, size_t chunkSize, size_t overlap )
{
    return {
        { totalSize, chunkSize, overlap, 0 },
        { totalSize, chunkSize, overlap, chunkCount( totalSize, chunkSize, overlap ) },
    };
}

} // namespace MR
