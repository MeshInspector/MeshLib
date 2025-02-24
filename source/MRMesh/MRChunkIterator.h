#pragma once

#include "MRMeshFwd.h"
#include "MRIteratorRange.h"

#include <iterator>

namespace MR
{

/// ...
struct Chunk
{
    /// ...
    size_t index;
    /// ...
    size_t offset;
    /// ...
    size_t size;
};

struct ChunkIterator
{
    size_t totalSize{ 0 };
    size_t chunkSize{ 0 };
    size_t overlap{ 0 };
    size_t index{ 0 };

    auto operator <=>( const ChunkIterator& ) const = default;

    using iterator_category = std::input_iterator_tag;
    using value_type = Chunk;
    using difference_type = std::ptrdiff_t;

    MRMESH_API ChunkIterator& operator ++();
    MRMESH_API Chunk operator *() const;
};

/// ...
MRMESH_API size_t chunkCount( size_t totalSize, size_t chunkSize, size_t overlap = 0 );

/// ...
MRMESH_API IteratorRange<ChunkIterator> splitByChunks( size_t totalSize, size_t chunkSize, size_t overlap = 0 );

} // namespace MR
