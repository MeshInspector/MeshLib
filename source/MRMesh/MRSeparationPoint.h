#pragma once

#include "MRId.h"
#include "MRVector3.h"
#include "MRVector.h"
#include "MRphmap.h"
#include <array>

namespace MR
{

enum class NeighborDir
{
    X, Y, Z, Count
};

using SeparationPointSet = std::array<VertId, size_t( NeighborDir::Count )>;
using SeparationPointMap = HashMap<size_t, SeparationPointSet>;

/// storage for points on voxel edges used in Marching Cubes algorithms
class SeparationPointStorage
{
public:
    struct alignas(64) Block
    {
        SeparationPointMap smap;
        std::vector<Vector3f> coords;
        /// after makeUniqueVids(), it is the unique id of first point in coords
        VertId shift;

        /// during filling, it is the id of next valid point;
        VertId nextVid() const { return VertId( coords.size() ); }

        Triangulation tris;
        Vector<VoxelId, FaceId> faceMap;
    };

    /// prepares storage for given number of blocks, each containing given size of voxels
    MRMESH_API void resize( size_t blockCount, size_t blockSize );

    /// get block for filling in the thread responsible for it
    Block & getBlock( size_t blockIndex ) { return blocks_[blockIndex]; }

    /// shifts vertex ids in each block (after they are filled) to make them unique;
    /// returns the total number of valid points in the storage
    MRMESH_API int makeUniqueVids();

    /// finds the set (locating the block) by voxel id
    auto findSeparationPointSet( size_t voxelId ) const -> const SeparationPointSet *
    {
        const auto & map = blocks_[voxelId / blockSize_].smap;
        auto it = map.find( voxelId );
        return ( it != map.end() ) ? &it->second : nullptr;
    }

    /// combines triangulations from every block into one and returns it
    MRMESH_API Triangulation getTriangulation( Vector<VoxelId, FaceId>* outVoxelPerFaceMap = nullptr ) const;

    /// obtains coordinates of all stored points
    MRMESH_API void getPoints( VertCoords & points ) const;

private:
    size_t blockSize_ = 0;
    std::vector<Block> blocks_;
};

} //namespace MR
