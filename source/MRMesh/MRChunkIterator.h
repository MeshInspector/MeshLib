#pragma once

#include "MRMeshFwd.h"
#include "MRIteratorRange.h"

#include <iterator>

namespace MR
{

/// array chunk representation
struct Chunk
{
    /// chunk offset
    size_t offset;
    /// chunk size; the last chunk's size may be smaller than other chunk's ones
    size_t size;
};

struct ChunkIterator
{
    size_t totalSize{ 0 };
    size_t chunkSize{ 0 };
    size_t overlap{ 0 };
    size_t index{ 0 };

    MRMESH_API bool operator ==( const ChunkIterator& other ) const;

    using iterator_category = std::input_iterator_tag;
    using value_type = Chunk;
    using difference_type = std::ptrdiff_t;
    using pointer = Chunk*;
    using reference = Chunk&;

    MRMESH_API ChunkIterator& operator ++();
    MRMESH_API ChunkIterator operator ++( int );

    MRMESH_API Chunk operator *() const;
    Chunk operator ->() const { return operator*(); }
};

/// returns the amount of chunks of given size required to cover the full array
MRMESH_API size_t chunkCount( size_t totalSize, size_t chunkSize, size_t overlap = 0 );

/// returns a pair of iterators for chunks covering the array of given size
MRMESH_API IteratorRange<ChunkIterator> splitByChunks( size_t totalSize, size_t chunkSize, size_t overlap = 0 );

} // namespace MR
