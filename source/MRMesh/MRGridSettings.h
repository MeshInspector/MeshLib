#pragma once

#include "MRVector2.h"
#include "MRBuffer.h"

namespace MR
{

/// settings defining regular grid, where each quadrangular cell is split on two triangles in one of two ways
struct GridSettings
{
    /// the number of cells in X and Y dimensions;
    /// the number of vertices will be at most (X+1)*(Y+1)
    Vector2i dim;

    /// grid coordinates to vertex Id; invalid vertex Id means that this vertex is missing in grid;
    /// index is x + y * ( settings.dim.x + 1 )
    BMap<VertId, size_t> vertIds;

    enum class EdgeType
    {
        Horizontal,  // (x,y) - (x+1,y)
        Vertical,    // (x,y) - (x,y+1)
        DiagonalA,   // (x,y) - (x+1,y+1)
        DiagonalB    // (x+1,y) - (x,y+1)
        // both DiagonalA and DiagonalB cannot return valid edges
    };
    /// grid coordinates of lower-left vertex and edge-type to edgeId with the origin in this vertex;
    /// both vertices of valid edge must be valid as well;
    /// index is 4 * ( x + y * ( settings.dim.x + 1 ) ) + edgeType
    BMap<UndirectedEdgeId, size_t> uedgeIds;

    enum class TriType
    {
        Lower, // (x,y), (x+1,y), (x+1,y+1) if DiagonalA or (x,y), (x+1,y), (x,y+1) if DiagonalB
        Upper  // (x,y), (x+1,y+1), (x,y+1) if DiagonalA or (x+1,y), (x+1,y+1), (x,y+1) if DiagonalB
    };
    /// grid coordinates of lower-left vertex and triangle-type to faceId;
    /// all 3 vertices and all 3 edges of valid face must be valid as well;
    /// index is 2 * ( x + y * settings.dim.x ) + triType
    BMap<FaceId, size_t> faceIds;
};

} // namespace MR
