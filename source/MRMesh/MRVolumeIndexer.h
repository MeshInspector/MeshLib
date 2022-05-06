#pragma once

#include "MRMeshFwd.h"
#include "MRVector3.h"
#include "MRId.h"

#include <array>

namespace MR
{

/// \defgroup VoxelGroup Voxel
/// \brief This chapter represents documentation about Volume (Voxel)
/// \{

/// \defgroup VolumeIndexerGroup Volume Indexer
/// \ingroup VoxelGroup
/// \{

enum class OutEdge : signed char
{
    Invalid = -1,
    PlusZ = 0,
    MinusZ,
    PlusY,
    MinusY,
    PlusX,
    MinusX,
    Count
};

static_assert( sizeof( OutEdge ) == 1 );

static const std::initializer_list<OutEdge> all6Edges = { OutEdge::PlusZ, OutEdge::MinusZ, OutEdge::PlusY, OutEdge::MinusY, OutEdge::PlusX, OutEdge::MinusX };

inline OutEdge opposite( OutEdge e )
{
    const static std::array<OutEdge, 7> map{ OutEdge::Invalid, OutEdge::MinusZ, OutEdge::PlusZ, OutEdge::MinusY, OutEdge::PlusY, OutEdge::MinusX, OutEdge::PlusX };
    return map[ (size_t)e + 1 ];
}

static constexpr int OutEdgeCount = 6;

class VolumeIndexer
{
public:
    VolumeIndexer( const Vector3i & dims );
    const Vector3i & dims() const { return dims_; }
    size_t size() const { return size_; }
    size_t sizeXY() const { return sizeXY_; }
    Vector3i toPos( VoxelId id ) const;
    VoxelId toVoxelId( const Vector3i & pos ) const;
    /// returns true if v1 is within at most 6 neighbors of v0
    bool areNeigbors( VoxelId v0, VoxelId v1 ) const { return areNeigbors( toPos( v0 ), toPos( v1 ) ); }
    bool areNeigbors( const Vector3i & pos0, const Vector3i & pos1 ) const { return ( pos0 - pos1 ).lengthSq() == 1; }
    /// returns id of v's neighbor specified by the edge
    VoxelId getNeighbor( VoxelId v, OutEdge toNei ) const { return getNeighbor( v, toPos( v ), toNei ); }
    MRMESH_API VoxelId getNeighbor( VoxelId v, const Vector3i & pos, OutEdge toNei ) const;

protected:
    Vector3i dims_;
    size_t sizeXY_ = 0; ///< = dims_.x * dims_.y
    size_t size_ = 0; ///< = dims_.x * dims_.y * dims_.z
};

inline VolumeIndexer::VolumeIndexer( const Vector3i & dims ) 
    : dims_( dims )
    , sizeXY_( size_t( dims_.x ) * dims_.y ) 
    , size_( sizeXY_ * dims_.z ) 
{ 
}

inline Vector3i VolumeIndexer::toPos( VoxelId id ) const
{
    assert( id.valid() );
    int z = int( id / sizeXY_ );
    int sumZ = int( id % sizeXY_ );
    int y = sumZ / dims_.x;
    int x = sumZ % dims_.x;
    return {x,y,z};
}

inline VoxelId VolumeIndexer::toVoxelId( const Vector3i & pos ) const
{
    return VoxelId{ pos.x + pos.y * size_t(dims_.x) + pos.z * sizeXY_ };
}

/// expands VoxelBitSet with given number of steps
MRMESH_API void expandVoxelsMask( VoxelBitSet& mask, const VolumeIndexer& indexer, int expansion = 1 );
/// shrinks VoxelBitSet with given number of steps
MRMESH_API void shrinkVoxelsMask( VoxelBitSet& mask, const VolumeIndexer& indexer, int shrinkage = 1 );

/// \}

} // namespace MR
