#pragma once

#include "MRMeshBuilderTypes.h"
#include "MRVector3.h"
#include "MRVector.h"
#include "MRphmap.h"
#include <array>
#include <cstring>

namespace MR
{

namespace MeshBuilder
{

using ThreePoints = std::array<Vector3f, 3>;

/// this makes bit-wise comparison of two Vector3f's thus making two NaNs equal
struct equalVector3f
{
    bool operator() ( const Vector3f & a, const Vector3f & b ) const
    {
        static_assert( sizeof( Vector3f ) == 12 );
        char ax[12], bx[12];
        std::memcpy( ax, &a, 12 );
        std::memcpy( bx, &b, 12 );
        return std::memcmp( ax, bx, 12 ) == 0;
    }
};

/// this class is responsible for giving a unique id to each vertex with distinct coordinates
class VertexIdentifier
{
public:
    /// prepare identification of vertices from given this number of triangles
    MRMESH_API void reserve( size_t numTris );
    /// identifies vertices from a chunk of triangles
    MRMESH_API void addTriangles( const std::vector<ThreePoints> & buffer );
    /// returns the number of triangles added so far
    size_t numTris() const { return tris_.size(); }
    /// obtains triangles with vertex ids
    std::vector<MeshBuilder::Triangle> takeTris() { return std::move( tris_ ); }
    /// obtains coordinates of unique points in the order of vertex ids
    VertCoords takePoints() { return std::move( points_ ); }

private:
    using VertInHMap = std::array<VertId*, 3>;
    std::vector<VertInHMap> vertsInHMap_;
    using HMap = phmap::parallel_flat_hash_map<Vector3f, VertId, phmap::priv::hash_default_hash<Vector3f>, equalVector3f>;
    HMap hmap_;
    std::vector<MeshBuilder::Triangle> tris_;
    VertCoords points_;
};

} //namespace MeshBuilder

} //namespace MR
