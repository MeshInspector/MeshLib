#pragma once

#include "MRId.h"
#include "MRVector3.h"
#include "MRphmap.h"
#include <array>

namespace MR
{

enum class NeighborDir
{
    X, Y, Z, Count
};

// point between two neighbor voxels
struct SeparationPoint
{
    Vector3f position; // coordinate
    VertId vid; // any valid VertId is ok
    // each SeparationPointMap element has three SeparationPoint, it is not guaranteed that all three are valid (at least one is)
    // so there are some points present in map that are not valid
    explicit operator bool() const
    {
        return vid.valid();
    }
};

using SeparationPointSet = std::array<SeparationPoint, size_t( NeighborDir::Count )>;
using SeparationPointMap = HashMap<size_t, SeparationPointSet>;

/// storage for points on voxel edges used in Marching Cubes algorithms
class SeparationPointStorage
{
public:
    /// prepares storage for given number of blocks, each containing given size of voxels
    explicit SeparationPointStorage( size_t blockCount, size_t blockSize );

    /// get block for filling in the thread responsible for it
    SeparationPointMap & getBlock( size_t blockIndex ) { return hmaps_[blockIndex]; }

    /// shifts vertex ids in each block (after they are filled) to make them unique
    void shiftVertIds( const std::function<int(size_t)> & getVertIndexShiftForVoxelId );

    /// finds the set (locating the block) by voxel id
    auto findSeparationPointSet( size_t voxelId ) const -> const SeparationPointSet *
    {
        const auto & map = hmaps_[voxelId / blockSize_];
        auto it = map.find( voxelId );
        return ( it != map.end() ) ? &it->second : nullptr;
    }

    /// obtains coordinates of all stored points
    void getPoints( VertCoords & points ) const;

private:
    size_t blockSize_ = 0;
    std::vector<SeparationPointMap> hmaps_;
};

} //namespace MR
