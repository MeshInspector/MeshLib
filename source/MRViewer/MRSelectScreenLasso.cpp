#include "MRSelectScreenLasso.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRObjectMesh.h"
#include "MRMesh/MRObjectPoints.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRRegionBoundary.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRVector2.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MR2to3.h"
#include "MRMesh/MRPolyline2.h"
#include "MRMesh/MRPolyline2Intersect.h"

#include <tbb/enumerable_thread_specific.h>

namespace MR
{

void SelectScreenLasso::addPoint( int mouseX, int mouseY )
{
    float mx = float( mouseX );
    float my = float( mouseY );
    if ( screenLoop_.empty() || screenLoop_.back().x != mx || screenLoop_.back().y != my )
    {
        screenLoop_.push_back( { mx, my } );
    }
}

std::vector<BitSet> SelectScreenLasso::calculateSelectedPixels( Viewer* viewer )
{
    if ( screenLoop_.empty() )
        return {};

    Viewer& viewerRef = *viewer;

    const auto& vpRect = viewerRef.viewport().getViewportRect();

    // convert polygon
    Contour2f contour( screenLoop_.size() + 1 );
    
    auto viewportId = viewerRef.viewport().id;
    for ( int i = 0; i < screenLoop_.size(); i++ )
        contour[i] = to2dim( viewerRef.screenToViewport( { screenLoop_[i].x, screenLoop_[i].y,0.f }, viewportId ) );
    contour.back() = contour.front();

    Polyline2 polygon( { std::move( contour ) } );
    // initialize line wise bitsets
    std::vector<BitSet> bsVec( int( height( vpRect ) ) + 1 );
    for ( auto& bs : bsVec )
        bs.resize( int( width( vpRect ) + 1 ) );

    auto box = Box2i( polygon.getBoundingBox() );
    box.min -= Vector2i::diagonal( 1 );
    box.max += Vector2i::diagonal( 1 );
    if ( box.min.x < 0 )
        box.min.x = 0;
    if ( box.min.y < 0 )
        box.min.y = 0;
    if ( box.max.x > int( width( vpRect ) + 1 ) )
        box.max.x = int( width( vpRect ) + 1 );
    if ( box.max.y > int( height( vpRect ) + 1 ) )
        box.max.y = int( height( vpRect ) + 1 );

    // mark all pixels in the polygon
    tbb::parallel_for( tbb::blocked_range<int>( box.min.y, box.max.y ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        for ( int y = range.begin(); y < range.end(); ++y )
        {
            auto& curBs = bsVec[y];
            for ( int x = box.min.x; x < box.max.x; x++ )
            {
                curBs.set( x, isPointInsidePolyline( polygon, Vector2f( float( x ), float( y ) ) ) );
            }
        }
    } );

    return bsVec;
}

FaceBitSet findIncidentFaces( const Viewport& viewport, const std::vector<BitSet>& bsVec, const ObjectMesh& obj,
                              bool onlyVisible, bool includeBackfaces, const std::vector<ObjectMesh*> * occludingMeshes )
{
    if ( bsVec.empty() )
        return {};

    bool any = false;
    for ( const auto& bs : bsVec )
        any = any || bs.any();
    if ( !any )
        return {};

    const auto& mesh = obj.mesh();
    const auto& vpRect = viewport.getViewportRect();
    const auto xf = obj.worldXf();

    auto toClipSpace = [&]( const Vector3f & meshPoint )
    {
        const auto p = xf( meshPoint );
        return viewport.projectToClipSpace( p );
    };

    auto inSelectedArea = [&]( const Vector3f & clipSpacePoint )
    {
        if ( clipSpacePoint[0] < -1.f || clipSpacePoint[0] > 1.f || clipSpacePoint[1] < -1.f || clipSpacePoint[1] > 1.f )
            return false;
        auto x = std::lround( ( clipSpacePoint[0] / 2.f + 0.5f ) * width( vpRect ) );
        auto y = std::lround( ( -clipSpacePoint[1] / 2.f + 0.5f ) * height( vpRect ) );
        return bsVec[y].test( x );
    };

    // find all verts inside
    auto verts = mesh->topology.getValidVerts();
    BitSetParallelFor( verts, [&] ( VertId i )
    {
        if ( !inSelectedArea( toClipSpace( mesh->points[i] ) ) )
            verts.set( i, false );
    } );

    struct Fragment 
    {
        Vector3f p;
        FaceId f;
    };

    const auto cameraEye = viewport.getCameraPoint();
    const Box2f clipArea( { -1.f, -1.f }, { 1.f, 1.f } );
    tbb::enumerable_thread_specific<std::vector<Fragment>> tlsFragments;
    BitSetParallelFor( mesh->topology.getValidFaces(), [&]( FaceId f )
    {
        Vector3f v[3];
        mesh->getTriPoints( f, v );
        if ( !includeBackfaces )
        {
            const auto n = cross( v[1] - v[0], v[2] - v[0] );
            if ( dot( xf.A * n, cameraEye ) < 0 )
                return;
        }
        Vector3f clip[3];
        Box2f triClipBox;
        for ( int i = 0; i < 3; ++i )
        {
            clip[i] = toClipSpace( v[i] );
            triClipBox.include( Vector2f{ clip[i].x, clip[i].y } );
        }
        triClipBox.intersect( clipArea );
        if ( !triClipBox.valid() )
            return;
        const int maxPixelSpan = ( int ) std::lround( std::max( triClipBox.size().x * width( vpRect ), triClipBox.size().y * height( vpRect ) ) / 2 );
        if ( maxPixelSpan < 6 ) // subdivide only triangles larger than 6 pixels in one of dimensions
            return;
        const int steps = std::min( maxPixelSpan / 2, 64 ); // sample over every second pixel
        const float rsteps = 1.0f / steps;
        // no samples on edges
        auto & fragments = tlsFragments.local();
        for ( int ia = 1; ia < steps; ++ia )
            for ( int ib = 1; ia + ib < steps; ++ib )
            {
                int ic = steps - ia - ib;
                const Vector3f cl = ia * rsteps * clip[0] + ib * rsteps * clip[1] + ic * rsteps * clip[2];
                if ( !inSelectedArea( cl ) )
                    continue;
                const Vector3f p = ia * rsteps * v[0] + ib * rsteps * v[1] + ic * rsteps * v[2];
                fragments.push_back( { .p = p, .f = f } );
            }
    } );

    std::vector<Fragment> largeTriFragments;
    size_t totalFrags = 0;
    for ( auto & fragments : tlsFragments )
        totalFrags += fragments.size();
    largeTriFragments.reserve( totalFrags );
    for ( auto & fragments : tlsFragments )
        largeTriFragments.insert( largeTriFragments.end(), fragments.begin(), fragments.end() );

    if ( onlyVisible )
    {
        std::vector<AffineXf3f> xfMeshToOccMesh;
        std::vector<Vector3f> cameraEyes; //relative to mesh
        std::vector<Line3fMesh> lineMeshes;
        xfMeshToOccMesh.emplace_back();
        cameraEyes.push_back( xf.inverse()( viewport.getCameraPoint() ) );
        lineMeshes.push_back( Line3fMesh{ .mesh = mesh.get(), .tree = &mesh->getAABBTree() } );
        if ( occludingMeshes )
        {
            for ( const auto * occ : *occludingMeshes )
            {
                if ( !occ || occ == &obj )
                    continue;
                const auto worldToOccMesh = occ->worldXf().inverse();
                xfMeshToOccMesh.push_back( worldToOccMesh * xf );
                cameraEyes.push_back( worldToOccMesh( viewport.getCameraPoint() ) );
                const auto * occmesh = occ->mesh().get();
                lineMeshes.push_back( Line3fMesh{ .mesh = occmesh, .tree = &occmesh->getAABBTree() } );
            }
        }

        tbb::enumerable_thread_specific<std::vector<Line3fMesh>> tlsLineMeshes( lineMeshes );
        auto isPointHidden = [&]( const Vector3f& point )
        {
            auto & myLineMeshes = tlsLineMeshes.local();
            assert( myLineMeshes.size() == cameraEyes.size() );
            for ( int i = 0; i < myLineMeshes.size(); ++i )
            {
                auto pointInOcc = xfMeshToOccMesh[i]( point );
                myLineMeshes[i].line = Line3f{ pointInOcc, cameraEyes[i] - pointInOcc };
            }
            return rayMultiMeshAnyIntersect( myLineMeshes, 0.0f, FLT_MAX );
        };

        BitSetParallelFor( verts, [&] ( VertId vid )
        {
            if ( isPointHidden( mesh->points[vid] ) )
                verts.set( vid, false );
        } );

        tbb::parallel_for( tbb::blocked_range<size_t>( size_t(0), largeTriFragments.size() ),
            [&] ( const tbb::blocked_range<size_t>& range )
        {
            for ( size_t i = range.begin(); i < range.end(); ++i )
            {
                auto & frag = largeTriFragments[i];
                if ( isPointHidden( frag.p ) )
                    frag.f = FaceId{}; //invalidate
            }
        } );
    }

    auto res = getIncidentFaces( mesh->topology, verts );
    if ( !includeBackfaces )
    {
        BitSetParallelFor( res, [&] ( FaceId i )
        {
            const auto& n = mesh->dirDblArea( i ); // non-unit norm (unnormalized)
            if ( dot( xf.A * n, cameraEye ) < 0.f )
                res.set( i, false );
        } );
    }
    for ( auto & frag : largeTriFragments )
        if ( frag.f )
            res.set( frag.f );
    return res;
}

VertBitSet findVertsInViewportArea( const Viewport& viewport, const std::vector<BitSet>& bsVec, const ObjectPoints& obj,
                                    bool includeBackfaces /*= true */ )
{
    if ( bsVec.empty() )
        return {};

    bool any = false;
    for ( const auto& bs : bsVec )
        any = any || bs.any();
    if ( !any )
        return {};

    const auto& pointCloud = obj.pointCloud();
    const auto& vpRect = viewport.getViewportRect();
    const auto xf = obj.worldXf();

    auto toClipSpace = [&]( const Vector3f& objPoint )
    {
        const auto p = xf( objPoint );
        return viewport.projectToClipSpace( p );
    };

    auto inSelectedArea = [&]( const Vector3f& clipSpacePoint )
    {
        if ( clipSpacePoint[0] < -1.f || clipSpacePoint[0] > 1.f || clipSpacePoint[1] < -1.f || clipSpacePoint[1] > 1.f )
            return false;
        auto x = std::lround( ( clipSpacePoint[0] / 2.f + 0.5f ) * width( vpRect ) );
        auto y = std::lround( ( -clipSpacePoint[1] / 2.f + 0.5f ) * height( vpRect ) );
        return bsVec[y].test( x );
    };

    // find all verts inside
    auto verts = pointCloud->validPoints;
    const auto cameraEye = viewport.getCameraPoint();
    const auto& normals = pointCloud->normals;
    const bool excludeBackface = !includeBackfaces && normals.size() >= pointCloud->points.size();
    BitSetParallelFor( verts, [&]( VertId i )
    {
        if ( !inSelectedArea( toClipSpace( pointCloud->points[i] ) ) || ( excludeBackface && dot( normals[i], cameraEye ) < 0 ) )
            verts.set( i, false );
    } );

    return verts;
}

}
