#include "MRRadiusCompensation2.h"
#include "MRPointsInBox.h"
#include "MRAABBTreePoints.h"
#include "MRMesh.h"
#include "MR2to3.h"

namespace MR
{

VertBitSet findVerticesInsideTool( const Mesh& mesh, const SphericalMillingCutter& tool )
{
    VertBitSet res;

    const auto & tree = mesh.getAABBTreePoints();
    auto meshBox = tree.getBoundingBox();
    if ( !meshBox.valid() )
        return res;

    Box3f toolBox;
    toolBox.min.x = tool.center.x - tool.radius;
    toolBox.min.y = tool.center.y - tool.radius;
    toolBox.min.z = meshBox.min.z;

    toolBox.max.x = tool.center.x + tool.radius;
    toolBox.max.y = tool.center.y + tool.radius;
    toolBox.max.z = tool.center.z + tool.radius;

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

} //namespace MR
