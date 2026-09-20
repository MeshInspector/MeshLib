#pragma once
#include "MRDelaunayTriangulationXY.h"

namespace MR
{

[[deprecated( "Use delaunayTriangulationXY( points, cb )" )]]
MR_BIND_IGNORE inline Expected<Mesh> terrainTriangulation( std::vector<Vector3f> points, const ProgressCallback& cb = {} )
{
    return delaunayTriangulationXY( std::move( points ), cb );
}

}
