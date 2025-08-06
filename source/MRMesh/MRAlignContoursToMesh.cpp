#include "MRAlignContoursToMesh.h"
#include "MR2DContoursTriangulation.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRMeshMetrics.h"
#include "MRMeshFillHole.h"
#include "MRAffineXf3.h"
#include "MRQuaternion.h"
#include "MRMeshIntersect.h"
#include "MRBox.h"

namespace MR
{

void addBaseToPlanarMesh( Mesh& mesh, float zOffset )
{
    MR_TIMER;
    mesh.pack(); // for some hard fonts with duplicated points (if triangulated contours have same points, duplicates are not used)
    // it's important to have all vertices valid:
    // first half is upper points of text and second half is lower points of text

    Mesh mesh2 = mesh;
    for ( auto& p : mesh2.points )
        p.z += zOffset;

    mesh2.topology.flipOrientation();

    mesh.addMesh( mesh2 );

    auto edges = mesh.topology.findHoleRepresentiveEdges();
    for ( int bi = 0; bi < edges.size() / 2; ++bi )
    {
        StitchHolesParams stitchParams;
        stitchParams.metric = getVerticalStitchMetric( mesh, Vector3f::plusZ() );
        buildCylinderBetweenTwoHoles( mesh, edges[bi], edges[edges.size() / 2 + bi], stitchParams );
    }
}

Expected<Mesh> alignContoursToMesh( const Mesh& mesh, const Contours2f& contours, const ContoursMeshAlignParams& params )
{
    auto contoursMesh = PlanarTriangulation::triangulateContours( contours );
    auto bbox = contoursMesh.computeBoundingBox();
    if ( !bbox.valid() )
        return unexpected( "Contours mesh is empty" );

    const float cStartDepth = bbox.diagonal() * 0.05f; // use relative depth to avoid floating errors
    addBaseToPlanarMesh( contoursMesh, -cStartDepth );

    auto diagonal = bbox.size(); diagonal.z = cStartDepth;
    AffineXf3f transform;

    const auto& vecx = params.xDirection.normalized();
    const auto norm = params.zDirection != nullptr ? *params.zDirection : mesh.pseudonormal( params.meshPoint );
    const auto vecy = cross( vecx, -norm ).normalized();

    const Vector3f pivotCoord{ bbox.min.x + diagonal.x * params.pivotPoint.x,
                               bbox.min.y + diagonal.y * params.pivotPoint.y,
                               0.0f };

    auto rotQ = Quaternionf( Vector3f::plusX(), vecx );
    // handle degenerated case
    auto newY = rotQ( Vector3f::plusY() );
    auto dotY = dot( newY, vecy );
    if ( std::abs( std::abs( dotY ) - 1.0f ) < 10.0f * std::numeric_limits<float>::epsilon() )
    {
        if ( dotY < 0.0f )
            rotQ = Quaternionf( vecx, PI_F ) * rotQ;
    }
    else
        rotQ = Quaternionf( newY, vecy ) * rotQ;
    AffineXf3f rot = AffineXf3f::linear( rotQ );

    auto translation = mesh.triPoint( params.meshPoint );

    transform =
        AffineXf3f::translation( translation ) *
        rot
        * AffineXf3f::translation( -pivotCoord );

    auto& contoursMeshPoints = contoursMesh.points;
    for ( auto& p : contoursMeshPoints )
        p = transform( p );

    auto plusOffsetDir = norm * std::abs( params.extrusion );
    auto minusOffsetDir = norm * ( cStartDepth - std::abs( params.extrusion ) );
    const auto maxMovement = std::max( 0.0f, params.maximumShift );
    for ( int i = 0; i < contoursMeshPoints.size() / 2; ++i )
    {
        PointOnFace hit;
        auto inter = rayMeshIntersect( mesh, Line3f{ contoursMeshPoints[VertId( i )] + norm * bbox.size().y, -norm } );
        if ( !inter )
            return unexpected( std::string( "Cannot align contours" ) );
        hit = inter.proj;

        auto coords = hit.point;
        auto dir = coords - contoursMeshPoints[VertId( i )];
        auto movement = dir.length();
        if ( movement > maxMovement )
            dir = ( maxMovement / movement ) * dir;

        contoursMeshPoints[VertId( i )] += dir + plusOffsetDir;
        contoursMeshPoints[VertId( i + contoursMeshPoints.size() / 2 )] += dir + minusOffsetDir;
    }
    return contoursMesh;

}

}