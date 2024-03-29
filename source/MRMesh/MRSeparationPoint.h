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
    struct Block
    {
        SeparationPointMap smap;
        VertId nextVid{ 0 };
    };

    /// prepares storage for given number of blocks, each containing given size of voxels
    explicit SeparationPointStorage( size_t blockCount, size_t blockSize );

    /// get block for filling in the thread responsible for it
    Block & getBlock( size_t blockIndex ) { return blocks_[blockIndex]; }

    /// shifts vertex ids in each block (after they are filled) to make them unique;
    /// returns the total number of valid points in the storage
    int makeUniqueVids();

    /// finds the set (locating the block) by voxel id
    auto findSeparationPointSet( size_t voxelId ) const -> const SeparationPointSet *
    {
        const auto & map = blocks_[voxelId / blockSize_].smap;
        auto it = map.find( voxelId );
        return ( it != map.end() ) ? &it->second : nullptr;
    }

    /// obtains coordinates of all stored points
    void getPoints( VertCoords & points ) const;

private:
    size_t blockSize_ = 0;
    std::vector<Block> blocks_;
};

} //namespace MR
