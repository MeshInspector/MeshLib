#include "MRRadiusCompensation2.h"
#include "MRPointsInBox.h"
#include "MRAABBTreePoints.h"
#include "MRMesh.h"
#include "MR2to3.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

namespace
{

Box3f getToolBox( const SphericalMillingCutter& tool, const Box3f& meshBox )
{
    Box3f toolBox;
    toolBox.min.x = tool.center.x - tool.radius;
    toolBox.min.y = tool.center.y - tool.radius;
    toolBox.min.z = meshBox.min.z;

    toolBox.max.x = tool.center.x + tool.radius;
    toolBox.max.y = tool.center.y + tool.radius;
    toolBox.max.z = tool.center.z + tool.radius;
    return toolBox;
}

} // anonymous namespace

VertBitSet findVerticesInsideTool( const Mesh& mesh, const SphericalMillingCutter& tool )
{
    VertBitSet res;

    const auto & tree = mesh.getAABBTreePoints();
    auto meshBox = tree.getBoundingBox();
    if ( !meshBox.valid() )
        return res;

    Box3f toolBox = getToolBox( tool, meshBox );
    findPointsInBox( tree, toolBox, [&]( VertId v, const Vector3f& p )
    {
        float distSq = 0;
        if ( p.z >= tool.center.z )
            distSq = distanceSq( p, tool.center );
        else
            distSq = distanceSq( to2dim( p ), to2dim( tool.center ) );
        if ( distSq < sqr( tool.radius ) )
            res.autoResizeSet( v );
    } );

    return res;
}

VertCoords compensateRadius2( const Mesh& mesh, float toolRadius )
{
    MR_TIMER

    const auto & tree = mesh.getAABBTreePoints();
    auto meshBox = tree.getBoundingBox();
    VertCoords res = mesh.points;
    if ( !meshBox.valid() )
        return res;

    VertCoords sumShifts( mesh.points.size() );
    Vector<int, VertId> numShifts( mesh.points.size() );

    for ( auto v : mesh.topology.getValidVerts() )
    {
        auto p = mesh.points[v];
        auto n = mesh.pseudonormal( v );
        SphericalMillingCutter tool
        {
            .center = p - toolRadius * n,
            .radius = toolRadius
        };
        Box3f toolBox = getToolBox( tool, meshBox );
        findPointsInBox( tree, toolBox, [&]( VertId vi, const Vector3f& p )
        {
            Vector3f delta = p - tool.center;
            if ( p.z < tool.center.z )
                delta.z = 0;

            auto distSq = delta.lengthSq();
            if ( distSq <= 0 || distSq >= sqr( toolRadius ) )
                return;
            auto dist = std::sqrt( distSq );
            Vector3f shift = delta / dist * ( toolRadius - dist );
            sumShifts[vi] += shift;
            ++numShifts[vi];
        } );
    }

    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        if ( numShifts[v] > 0 )
            res[v] += sumShifts[v] / float( numShifts[v] );
    } );

    return res;
}

} //namespace MR
