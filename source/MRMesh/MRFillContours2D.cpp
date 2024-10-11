#include "MRFillContours2D.h"
#include "MRMesh.h"
#include "MRVector2.h"
#include "MR2DContoursTriangulation.h"
#include "MRRingIterator.h"
#include "MREdgePaths.h"
#include "MRAffineXf3.h"
#include "MRTimer.h"
#include "MRRegionBoundary.h"
#include "MRMakeSphereMesh.h"
#include "MRMeshTrimWithPlane.h"
#include "MRPlane3.h"
#include "MRGTest.h"
#include <limits>
#include "MRMeshSave.h"
#include "MRFillContour.h"

namespace MR
{

class FromOxyPlaneCalculator
{
public:
    void addLineSegm( const Vector3d & a, const Vector3d & b )
    {
        sumPts_ += a;
        sumPts_ += b;
        numPts_ += 2;
        sumCross_ += cross( a, b );
    }
    void addLineSegm( const Vector3f & a, const Vector3f & b )
    {
        addLineSegm( Vector3d( a ), Vector3d( b ) );
    }
    AffineXf3d getXf() const
    {
        if ( numPts_ <= 0 )
            return {};
        auto planeNormal = sumCross_.normalized();
        auto center = sumPts_ / double( numPts_ );
        return { Matrix3d::rotation( Vector3d::plusZ(), planeNormal ), center };
    }

private:
    Vector3d sumPts_;
    Vector3d sumCross_;
    int numPts_ = 0;
};

AffineXf3f getXfFromOxyPlane( const Mesh& mesh, const std::vector<EdgePath>& paths )
{
    FromOxyPlaneCalculator c;
    for ( const auto& path : paths )
    {
        for ( const auto& edge : path )
            c.addLineSegm( mesh.orgPnt( edge ), mesh.destPnt( edge ) );
    }
    return AffineXf3f( c.getXf() );
}

AffineXf3f getXfFromOxyPlane( const Contours3f& contours )
{
    FromOxyPlaneCalculator c;
    for ( const auto& contour : contours )
    {
        for ( int i = 0; i + 1 < contour.size(); ++i )
            c.addLineSegm( contour[i], contour[i + 1] );
    }
    return AffineXf3f( c.getXf() );
}

Expected<void> fillContours2D( Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges )
{
    MR_TIMER
    // check input
    assert( !holeRepresentativeEdges.empty() );
    if ( holeRepresentativeEdges.empty() )
        return unexpected( "No hole edges are given" );

    // reorder to make edges ring with hole on left side
    bool badEdge = false;
    auto& meshTopology = mesh.topology;
    for ( const auto& edge : holeRepresentativeEdges )
    {
        if ( meshTopology.left( edge ) )
        {
            badEdge = true;
            break;
        }
    }
    assert( !badEdge );
    if ( badEdge )
        return unexpected( "Some hole edges have left face" );

    // make border rings
    std::vector<EdgeLoop> paths( holeRepresentativeEdges.size() );
    for ( int i = 0; i < paths.size(); ++i )
        paths[i] = trackRightBoundaryLoop( meshTopology, holeRepresentativeEdges[i] );

    // find transformation from world to plane space and back
    const auto planeXf = getXfFromOxyPlane( mesh, paths );
    const auto planeXfInv = planeXf.inverse();

    // make contours2D (on plane) from border rings (in world)
    Contours2f contours2f;
    contours2f.reserve( paths.size() );
    for ( const auto& path : paths )
    {
        contours2f.emplace_back();
        auto& contour = contours2f.back();
        contour.reserve( path.size() + 1 );
        for ( const auto& edge : path )
        {
            const auto localPoint = planeXfInv( mesh.orgPnt( edge ) );
            contour.emplace_back( Vector2f( localPoint.x, localPoint.y ) );
        }
        contour.emplace_back( contour.front() );
    }

    auto holeVertIds = std::make_unique<PlanarTriangulation::HolesVertIds>(
        PlanarTriangulation::findHoleVertIdsByHoleEdges( mesh.topology, paths ) );

    std::vector<EdgePath> newPaths;
    // make patch surface
    auto fillResult = PlanarTriangulation::triangulateDisjointContours( contours2f, holeVertIds.get(), &newPaths );
    holeVertIds.reset();
    if ( !fillResult )
        return unexpected( "Cannot triangulate contours with self-intersections" );
    Mesh& patchMesh = *fillResult;

    // transform patch surface from plane to world space
    auto& patchMeshPoints = patchMesh.points;
    for ( auto& point : patchMeshPoints )
        point = planeXf( point );

    if ( paths.size() != newPaths.size() )
        return unexpected( "Patch surface borders size different from original mesh borders size" );

    std::vector<EdgePath> invertedHoles;
    invertedHoles.reserve( newPaths.size() );
    for ( int i = 0; i < paths.size(); ++i )
    {
        if ( paths[i].size() != newPaths[i].size() )
            return unexpected( "Patch surface borders size different from original mesh borders size" );

        // degenerate holes might invert sometimes (it is expected as far as planar triangulation does not now about input topology)
        if ( newPaths[i].empty() || patchMesh.topology.right( newPaths[i].front() ) )
            if ( !newPaths[i].empty() )
                MR::reverse( invertedHoles.emplace_back( newPaths[i] ) );
    }
    if ( !invertedHoles.empty() )
    {
        auto invertedParts = fillContourLeft( patchMesh.topology, invertedHoles );
        auto invertedEdges = getIncidentEdges( patchMesh.topology, invertedParts );
        patchMesh.topology.flipOrientation( &invertedEdges );

        // validate one more time
        for ( int i = 0; i < paths.size(); ++i )
            if ( newPaths[i].empty() || patchMesh.topology.right( newPaths[i].front() ) )
                if ( !newPaths[i].empty() )
                    return unexpected( "Patch surface borders are incompatible with mesh borders" );
    }

    // move patch surface border points to original position (according original mesh)
    auto& patchMeshTopology = patchMesh.topology;
    auto& meshPoints = mesh.points;
    for ( int i = 0; i < paths.size(); ++i )
    {
        auto& path = paths[i];
        auto& newPath = newPaths[i];
        for ( int j = 0; j < path.size(); ++j )
            patchMeshPoints[patchMeshTopology.org( newPath[j] )] = meshPoints[meshTopology.org( path[j] )];
    }
    
    // add patch surface to original mesh
    mesh.addPartByMask( patchMesh, patchMesh.topology.getValidFaces(), false, paths, newPaths );
    return {};
}

TEST( MRMesh, fillContours2D )
{
    Mesh sphereBig = makeUVSphere( 1.0f, 32, 32 );
    Mesh sphereSmall = makeUVSphere( 0.7f, 16, 16 );

    sphereSmall.topology.flipOrientation();
    sphereBig.addPart( std::move( sphereSmall ) );

    trimWithPlane( sphereBig, TrimWithPlaneParams{ .plane = Plane3f::fromDirAndPt( Vector3f::plusZ(), Vector3f() ) } );
    sphereBig.pack();

    auto firstNewFace = sphereBig.topology.lastValidFace() + 1;
    fillContours2D( sphereBig, sphereBig.topology.findHoleRepresentiveEdges() );
    for ( FaceId f = firstNewFace; f <= sphereBig.topology.lastValidFace(); ++f )
    {   
        EXPECT_TRUE( std::abs( dot( sphereBig.dirDblArea( f ).normalized(), Vector3f::minusZ() ) - 1.0f ) < std::numeric_limits<float>::epsilon() );
    }
}

}
