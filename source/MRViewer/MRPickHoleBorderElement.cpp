#include "MRPickHoleBorderElement.h"
#include "MRViewer.h"
#include "MRViewport.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRRingIterator.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPolylineEdgeIterator.h"

namespace MR
{

// check that we see point (projected mouse position) on edge
bool isOnTheScreen( const std::shared_ptr<Object>& obj, const Vector3f& projPoint, const FaceId& faceId )
{
    Viewport& viewport = Viewer::instanceRef().viewport();

    Vector2f projPoint2f = Vector2f( projPoint.x, projPoint.y );
    auto pick = viewport.pickRenderObject( { .point = projPoint2f } );
    if ( pick.first && pick.first != obj )
        return false;
    auto clipPick = viewport.projectToViewportSpace( pick.second.point );
    return ( !pick.second.face.valid() ) || pick.second.face == faceId || ( clipPick.z - projPoint.z >= 0.f );
}
bool isOnTheScreen( const std::shared_ptr<Object>& obj, const Vector3f& projPoint, const UndirectedEdgeId& edgeId )
{
    return isOnTheScreen( obj, projPoint, FaceId( int( edgeId ) ) );
}

// find squared distance from point Q to line segment Segm
// if Q projection point doesn't correspond to Segm segment, distance to Segm.a or Segm.b will be returned
// Note: ignores Z coordinate
float findPixelDistSq( const Vector3f& q, const LineSegm3f& segm, Vector3f& projPoint, float& posOnEdge )
{
    auto dir = Vector2f( segm.b.x - segm.a.x, segm.b.y - segm.a.y );
    auto p1Q = Vector2f( q.x - segm.a.x, q.y - segm.a.y );
    auto lSq = dir.lengthSq();

    // if p2==p1
    if ( lSq == 0 )
        return ( q - segm.a ).lengthSq();

    posOnEdge = dot( p1Q, dir ) / lSq;

    // handle the case of out of segment projection
    posOnEdge = std::clamp( posOnEdge, 0.f, 1.f );

    projPoint = segm.a + posOnEdge * ( segm.b - segm.a );
    return Vector2f( q.x - projPoint.x, q.y - projPoint.y ).lengthSq();
}

HoleEdgePoint findClosestToMouseHoleEdge( const Vector2i& mousePos, const std::shared_ptr<ObjectMeshHolder>& objMesh,
                                          const std::vector<EdgeId>& holeRepresentativeEdges,
                                          float accuracy /*= 5.5f*/, bool attractToVert /*= false*/, float cornerAccuracy /*= 10.5f*/ )
{
    const Mesh& mesh = *objMesh->mesh();
    HoleEdgePoint result;
    Viewer& viewerRef = Viewer::instanceRef();
    Viewport& viewport = viewerRef.viewport();

    auto mousePix = Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f );
    mousePix = viewerRef.screenToViewport( mousePix, viewport.id );

    Vector3f projPointOut;
    float posOnEdge = 0.f;
    float minDistEdge = accuracy * accuracy;
    float minDistVert = cornerAccuracy * cornerAccuracy;

    auto xf = objMesh->worldXf();
    for ( int i = 0; i < holeRepresentativeEdges.size(); i++ )
    {
        const EdgeId& initEdge = holeRepresentativeEdges[i];
        for ( const auto& e : leftRing( mesh.topology, initEdge ) )
        {
            auto pOrg = xf(mesh.orgPnt( e ));
            auto orgPix = viewport.projectToViewportSpace( pOrg );
            auto pDest = xf(mesh.destPnt( e ));
            auto destPix = viewport.projectToViewportSpace( pDest );
            auto dist = findPixelDistSq( mousePix, { orgPix, destPix }, projPointOut, posOnEdge );
            if ( attractToVert )
            {
                if ( dist >= minDistVert )
                    continue;

                Vector3f vertPix;
                float pos = 0.f;
                if ( posOnEdge < 0.5 )
                    vertPix = orgPix;
                else
                {
                    vertPix = destPix;
                    pos = 1.f;
                }

                if ( ( vertPix - mousePix ).lengthSq() < minDistVert && isOnTheScreen( objMesh, vertPix, mesh.topology.right( e ) ) )
                {
                    minDistVert = dist;
                    result = { i, {e, pos} };
                }
                else if ( dist < minDistEdge && isOnTheScreen( objMesh, projPointOut, mesh.topology.right( e ) ) )
                {
                    minDistEdge = dist;
                    result = { i, {e, posOnEdge} };
                }

            }
            else if ( dist < minDistEdge && isOnTheScreen( objMesh, projPointOut, mesh.topology.right( e ) ) )
            {
                minDistEdge = dist;
                result = { i, {e, posOnEdge} };
            }
        }
    }


    return result;
}

HoleEdgePoint findClosestToMouseEdge( const Vector2i& mousePos, const std::vector<std::shared_ptr<ObjectLinesHolder>>& objsLines, float accuracy /*= 5.5f */ )
{
    Viewer& viewerRef = getViewerInstance();
    Viewport& viewport = viewerRef.viewport();

    auto mousePix = Vector3f( float( mousePos.x ), float( mousePos.y ), 0.f );
    mousePix = viewerRef.screenToViewport( mousePix, viewport.id );

    Vector3f projPointOut;
    float posOnEdge = 0.f;
    float minDistEdge = accuracy * accuracy;

    HoleEdgePoint result;

    for ( int i = 0; i < objsLines.size(); ++i )
    {
        const auto& objLines = objsLines[i];
        const Polyline3& polyline = *objLines->polyline();
        auto xf = objLines->worldXf();
        for ( auto ue : undirectedEdges( polyline.topology ) )
        {
            EdgeId e = ue;
            auto pOrg = xf( polyline.orgPnt( e ) );
            auto orgPix = viewport.projectToViewportSpace( pOrg );
            auto pDest = xf( polyline.destPnt( e ) );
            auto destPix = viewport.projectToViewportSpace( pDest );
            auto dist = findPixelDistSq( mousePix, { orgPix, destPix }, projPointOut, posOnEdge );
            if ( dist < minDistEdge && isOnTheScreen( objLines, projPointOut, ue ) )
            {
                minDistEdge = dist;
                result = { i, { EdgeId(size_t(ue)), posOnEdge } };
            }
        }
    }

    return result;
}

}
