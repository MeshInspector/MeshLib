#include "MRChunkIterator.h"

namespace MR
{

bool ChunkIterator::operator==( const ChunkIterator& other ) const
{
    return std::tie( totalSize, chunkSize, overlap, index )
        == std::tie( other.totalSize, other.chunkSize, other.overlap, other.index );
}

ChunkIterator& ChunkIterator::operator++()
{
    ++index;
    return *this;
}

ChunkIterator ChunkIterator::operator++( int )
{
    auto copy = *this;
    ++( *this );
    return copy;
}

Chunk ChunkIterator::operator*() const
{
    const auto offset = index * ( chunkSize - overlap );
    return {
        .index = index,
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
    return ( size + step - 1 ) / step;
}

IteratorRange<ChunkIterator> splitByChunks( size_t totalSize, size_t chunkSize, size_t overlap )
{
    return {
        { totalSize, chunkSize, overlap, 0 },
        { totalSize, chunkSize, overlap, chunkCount( totalSize, chunkSize, overlap ) },
    };
}

} // namespace MR
