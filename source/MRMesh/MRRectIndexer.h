#pragma once

#include "MRMeshFwd.h"
#include "MRVector2.h"
#include "MRId.h"
#include <array>

namespace MR
{

/// \addtogroup DistanceMapGroup
/// \{

enum class OutEdge2 : signed char
{
    Invalid = -1,
    PlusY,
    MinusY,
    PlusX,
    MinusX,
    Count
};

static_assert( sizeof( OutEdge2 ) == 1 );

static const std::initializer_list<OutEdge2> allOutEdges2 = { OutEdge2::PlusY, OutEdge2::MinusY, OutEdge2::PlusX, OutEdge2::MinusX };

inline OutEdge2 opposite( OutEdge2 e )
{
    const static std::array<OutEdge2, 5> map{ OutEdge2::Invalid, OutEdge2::MinusY, OutEdge2::PlusY, OutEdge2::MinusX, OutEdge2::PlusX };
    return map[ (size_t)e + 1 ];
}

static constexpr int OutEdge2Count = 4;

/// a class for converting 2D integer coordinates into 1D linear coordinates and backward
class RectIndexer
{
public:
    constexpr RectIndexer() noexcept = default;
    RectIndexer( const Vector2i & dims );
    void resize( const Vector2i & dims );
    const Vector2i & dims() const { return dims_; }
    size_t size() const { return size_; }
    Vector2i toPos( PixelId id ) const { assert( id.valid() ); return toPos( size_t( int( id ) ) ); }
    Vector2i toPos( size_t id ) const;
    PixelId toPixelId( const Vector2i & pos ) const { return PixelId{ int( toIndex( pos ) ) }; }
    size_t toIndex( const Vector2i & pos ) const;
    /// returns true if v1 is within at most 4 neighbors of v0
    bool areNeigbors( PixelId v0, PixelId v1 ) const { return areNeigbors( toPos( v0 ), toPos( v1 ) ); }
    bool areNeigbors( const Vector2i & pos0, const Vector2i & pos1 ) const { return ( pos0 - pos1 ).lengthSq() == 1; }
    /// returns id of v's neighbor specified by the edge
    PixelId getNeighbor( PixelId v, OutEdge2 toNei ) const { return getNeighbor( v, toPos( v ), toNei ); }
    MRMESH_API PixelId getNeighbor( PixelId v, const Vector2i & pos, OutEdge2 toNei ) const;

protected:
    Vector2i dims_;
    size_t size_ = 0; ///< = dims_.x * dims_.y
};

inline RectIndexer::RectIndexer( const Vector2i & dims )
{
    resize( dims );
}

inline void RectIndexer::resize( const Vector2i & dims ) 
{
    dims_ = dims;
    size_ = size_t( dims_.x ) * dims_.y;
}

inline Vector2i RectIndexer::toPos( size_t id ) const
{
    int y = int(id) / dims_.x;
    int x = int(id) % dims_.x;
    return {x, y};
}

inline size_t RectIndexer::toIndex( const Vector2i & pos ) const
{
    return pos.x + pos.y * size_t(dims_.x);
}

/// expands PixelBitSet with given number of steps
MRMESH_API void expandPixelMask( PixelBitSet& mask, const RectIndexer& indexer, int expansion = 1 );
/// shrinks PixelBitSet with given number of steps
MRMESH_API void shrinkPixelMask( PixelBitSet& mask, const RectIndexer& indexer, int shrinkage = 1 );

/// \}

} // namespace MR
