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

[[nodiscard]] inline OutEdge opposite( OutEdge e )
{
    const static std::array<OutEdge, 7> map{ OutEdge::Invalid, OutEdge::MinusZ, OutEdge::PlusZ, OutEdge::MinusY, OutEdge::PlusY, OutEdge::MinusX, OutEdge::PlusX };
    return map[ (size_t)e + 1 ];
}

static constexpr int OutEdgeCount = 6;

constexpr Vector3i neiPosDelta[ OutEdgeCount ] =
{
    { 0, 0, 1 }, // PlusZ
    { 0, 0,-1 }, // MinusZ
    { 0, 1, 0 }, // PlusY
    { 0,-1, 0 }, // MinusY
    { 1, 0, 0 }, // PlusX
    {-1, 0, 0 }  // MinusX
};

/// contains both linear Id and 3D coordinates of the same voxel
struct VoxelLocation
{
    VoxelId id;
    Vector3i pos;

    /// check for validity
    explicit operator bool() const { return id.valid(); }
};

class VolumeIndexer
{
public:
    [[nodiscard]] VolumeIndexer( const Vector3i & dims );

    [[nodiscard]] const Vector3i & dims() const { return dims_; }

    /// returns the total number of voxels
    [[nodiscard]] size_t size() const { return size_; }

    /// returns the last plus one voxel Id for defining iteration range
    [[nodiscard]] VoxelId endId() const { return VoxelId( size_ ); }

    [[nodiscard]] size_t sizeXY() const { return sizeXY_; }

    [[nodiscard]] Vector3i toPos( VoxelId id ) const;

    [[nodiscard]] VoxelId toVoxelId( const Vector3i & pos ) const;

    [[nodiscard]] VoxelLocation toLoc( VoxelId id ) const { return { id, toPos( id ) }; }
    [[nodiscard]] VoxelLocation toLoc( const Vector3i & pos ) const { return { toVoxelId( pos ), pos }; }

    /// returns true if this voxel is within dimensions
    [[nodiscard]] bool isInDims( const Vector3i& pos ) const { return pos.x >= 0 && pos.x < dims_.x && pos.y >= 0 && pos.y < dims_.y && pos.z >= 0 && pos.z < dims_.z; }

    /// returns true if this voxel is on the boundary of the volume
    [[nodiscard]] bool isBdVoxel( const Vector3i & pos ) const { return pos.x == 0 || pos.x + 1 == dims_.x || pos.y == 0 || pos.y + 1 == dims_.y || pos.z == 0 || pos.z + 1 == dims_.z; }

    /// returns true if v1 is within at most 6 neighbors of v0
    [[nodiscard]] bool areNeigbors( VoxelId v0, VoxelId v1 ) const { return areNeigbors( toPos( v0 ), toPos( v1 ) ); }
    [[nodiscard]] bool areNeigbors( const Vector3i & pos0, const Vector3i & pos1 ) const { return ( pos0 - pos1 ).lengthSq() == 1; }

    /// given existing voxel at (pos), returns whether it has valid neighbor specified by the edge (toNei)
    [[nodiscard]] MRMESH_API bool hasNeighbour( const Vector3i & pos, OutEdge toNei ) const;

    /// returns id of v's neighbor specified by the edge
    [[nodiscard]] VoxelId getNeighbor( VoxelId v, OutEdge toNei ) const { return getNeighbor( v, toPos( v ), toNei ); }
    [[nodiscard]] VoxelId getNeighbor( VoxelId v, const Vector3i & pos, OutEdge toNei ) const { return hasNeighbour( pos, toNei ) ? getExistingNeighbor( v, toNei ) : VoxelId{}; }

    /// given existing voxel at (loc), returns its neighbor specified by the edge (toNei);
    /// if the neighbour does not exist (loc is on boundary), returns invalid VoxelLocation
    [[nodiscard]] VoxelLocation getNeighbor( const VoxelLocation & loc, OutEdge toNei ) const;

    /// returns id of v's neighbor specified by the edge, which is known to exist (so skipping a lot of checks)
    [[nodiscard]] VoxelId getExistingNeighbor( VoxelId v, OutEdge toNei ) const;
    [[nodiscard]] VoxelId getNeighbor( VoxelId v, const Vector3i & pos, bool bdPos, OutEdge toNei ) const { return bdPos ? getNeighbor( v, pos, toNei ) : getExistingNeighbor( v, toNei ); }

protected:
    Vector3i dims_;
    size_t sizeXY_ = 0; ///< = dims_.x * dims_.y
    size_t size_ = 0; ///< = dims_.x * dims_.y * dims_.z
    int neiInc_[ OutEdgeCount ] = {};
};

inline VolumeIndexer::VolumeIndexer( const Vector3i & dims ) 
    : dims_( dims )
    , sizeXY_( size_t( dims_.x ) * dims_.y ) 
    , size_( sizeXY_ * dims_.z ) 
{ 
    neiInc_[(int)OutEdge::PlusZ]  = (int)sizeXY_;
    neiInc_[(int)OutEdge::MinusZ] =-(int)sizeXY_;
    neiInc_[(int)OutEdge::PlusY] =  dims_.x;
    neiInc_[(int)OutEdge::MinusY] =-dims_.x;
    neiInc_[(int)OutEdge::PlusX] =  1;
    neiInc_[(int)OutEdge::MinusX] =-1;
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

inline VoxelId VolumeIndexer::getExistingNeighbor( VoxelId v, OutEdge toNei ) const
{
    assert( toNei > OutEdge::Invalid && toNei < OutEdge::Count );
    return v + neiInc_[(int)toNei];
}

inline VoxelLocation VolumeIndexer::getNeighbor( const VoxelLocation & loc, OutEdge toNei ) const
{
    assert( loc );
    return hasNeighbour( loc.pos, toNei ) ?
        VoxelLocation{ getExistingNeighbor( loc.id, toNei ), loc.pos + neiPosDelta[(int)toNei] } :
        VoxelLocation{};
}

/// expands VoxelBitSet with given number of steps
MRMESH_API void expandVoxelsMask( VoxelBitSet& mask, const VolumeIndexer& indexer, int expansion = 1 );

/// shrinks VoxelBitSet with given number of steps
MRMESH_API void shrinkVoxelsMask( VoxelBitSet& mask, const VolumeIndexer& indexer, int shrinkage = 1 );

/// \}

} // namespace MR
