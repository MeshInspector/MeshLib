#include "MRSelectScreenLasso.h"
#include "MRViewport.h"
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
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPolyline2Intersect.h"
#include "MRMesh/MRPolylineProject.h"
#include "MRMesh/MRPointCloud.h"

namespace MR
{

BitSet calculateSelectedPixelsInsidePolygon( const Contour2f & screenPoints )
{
    if ( screenPoints.empty() )
        return {};

    Viewer& viewer = getViewerInstance();

    const auto& vpRect = viewer.viewport().getViewportRect();

    // convert polygon
    Contour2f contour( screenPoints.size() + 1 );
    
    auto viewportId = viewer.viewport().id;
    for ( int i = 0; i < screenPoints.size(); i++ )
        contour[i] = to2dim( viewer.screenToViewport( { screenPoints[i].x, screenPoints[i].y, 0.f }, viewportId ) );
    contour.back() = contour.front();

    Polyline2 polygon( { std::move( contour ) } );
    auto width = int( MR::width( vpRect ) );
    auto height = int( MR::height( vpRect ) );
    BitSet resBS( width * height );

    auto box = Box2i( polygon.getBoundingBox() );
    box.min -= Vector2i::diagonal( 1 );
    box.max += Vector2i::diagonal( 1 );
    if ( box.min.x < 0 )
        box.min.x = 0;
    if ( box.min.y < 0 )
        box.min.y = 0;
    if ( box.max.x >= width )
        box.max.x = width - 1;
    if ( box.max.y >= height )
        box.max.y = height - 1;

    // mark all pixels in the polygon
    BitSetParallelForAll( resBS, [&] ( size_t i )
    {
        Vector2i coord( int( i ) % width, int( i ) / width );
        if ( !box.contains( coord ) )
            return;
        resBS.set( i, isPointInsidePolyline( polygon, Vector2f( coord ) ) );
    } );

    return resBS;
}

BitSet calculateSelectedPixelsNearPolygon( const Contour2f & screenPoints, float radiusPix )
{
    if ( screenPoints.empty() )
        return {};

    Viewer& viewer = getViewerInstance();

    const auto& vpRect = viewer.viewport().getViewportRect();

    // convert polygon
    Contour2f contour( screenPoints.size() );

    auto viewportId = viewer.viewport().id;
    for ( int i = 0; i < screenPoints.size(); i++ )
        contour[i] = to2dim( viewer.screenToViewport( { screenPoints[i].x, screenPoints[i].y,0.f }, viewportId ) );
    if ( contour.size() == 1 )
        contour.emplace_back( contour.front() );

    Polyline2 polygon;
    polygon.addFromPoints( contour.data(), contour.size(), false );
    polygon.getAABBTree(); // create tree first

    auto width = int( MR::width( vpRect ) );
    auto height = int( MR::height( vpRect ) );
    BitSet resBS( width * height );

    auto radSq = radiusPix * radiusPix;
    // mark all pixels in the polygon
    BitSetParallelForAll( resBS, [&] ( size_t i )
    {
        Vector2i coord( int( i ) % width, int( i ) / width );
        auto projRes = findProjectionOnPolyline2( Vector2f( coord ), polygon, radSq );
        if ( projRes.line.valid() )
            resBS.set( i );
    } );

    return resBS;
}

FaceBitSet findIncidentFaces( const Viewport& viewport, const BitSet& pixBs, const ObjectMesh& obj,
                              bool onlyVisible, bool includeBackfaces, const std::vector<ObjectMesh*> * occludingMeshes )
{
    if ( pixBs.none() )
        return {};

    const auto& mesh = obj.mesh();
    const auto& vpRect = viewport.getViewportRect();
    const auto xf = obj.worldXf();

    auto toClipSpace = [&]( const Vector3f & meshPoint )
    {
        const auto p = xf( meshPoint );
        return viewport.projectToClipSpace( p );
    };

    auto width = MR::width( vpRect );
    auto height = MR::height( vpRect );

    auto inSelectedArea = [&]( const Vector3f & clipSpacePoint )
    {
        if ( clipSpacePoint[0] < -1.f || clipSpacePoint[0] > 1.f || clipSpacePoint[1] < -1.f || clipSpacePoint[1] > 1.f )
            return false;
        auto y = std::clamp( std::lround( ( -clipSpacePoint[1] / 2.f + 0.5f ) * height ), long( 0 ), long( height )-1 );
        auto x = std::clamp( std::lround( ( clipSpacePoint[0] / 2.f + 0.5f ) * width ), long( 0 ), long( width ) - 1 );
        return pixBs.test( x + long( width ) * y );
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

    const auto orthoBackwards = viewport.getBackwardDirection();
    const Box2f clipArea( { -1.f, -1.f }, { 1.f, 1.f } );
    tbb::enumerable_thread_specific<std::vector<Fragment>> tlsFragments;
    BitSetParallelFor( mesh->topology.getValidFaces(), [&]( FaceId f )
    {
        Vector3f v[3];
        mesh->getTriPoints( f, v );
        if ( !includeBackfaces )
        {
            const auto n = cross( v[1] - v[0], v[2] - v[0] );
            Vector3f cameraDir;
            if ( viewport.getParameters().orthographic )
                cameraDir = orthoBackwards;
            else
                cameraDir = -viewport.unprojectPixelRay( to2dim( viewport.projectToViewportSpace( mesh->triCenter( f ) ) ) ).d;
            if ( dot( xf.A * n, cameraDir ) < 0 )
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
        const int maxPixelSpan = ( int ) std::lround( std::max( triClipBox.size().x * width, triClipBox.size().y * height ) / 2 );
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

        tbb::enumerable_thread_specific<std::vector<Line3fMesh>> tlsLineMeshes( std::cref( lineMeshes ) );
        const auto& clippingPlane = viewport.getParameters().clippingPlane;
        const bool useClipping = obj.globalClippedByPlane( viewport.id );

        auto isPointHidden = [&]( const Vector3f& point ) -> bool
        {
            if ( useClipping && clippingPlane.distance( xf( point ) ) > 0 )
                return true;

            auto & myLineMeshes = tlsLineMeshes.local();
            assert( myLineMeshes.size() == cameraEyes.size() );
            for ( int i = 0; i < myLineMeshes.size(); ++i )
            {
                auto pointInOcc = xfMeshToOccMesh[i]( point );
                myLineMeshes[i].line = Line3f{ pointInOcc, cameraEyes[i] - pointInOcc };
            }
            return (bool)rayMultiMeshAnyIntersect( myLineMeshes, 0.0f, FLT_MAX );
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
        BitSetParallelFor( res, [&] ( FaceId f )
        {
            const auto& n = mesh->dirDblArea( f ); // non-unit norm (unnormalized)
            Vector3f cameraDir;
            if ( viewport.getParameters().orthographic )
                cameraDir = orthoBackwards;
            else
                cameraDir = -viewport.unprojectPixelRay( to2dim( viewport.projectToViewportSpace( mesh->triCenter( f ) ) ) ).d;
            if ( dot( xf.A * n, cameraDir ) < 0.f )
                res.set( f, false );
        } );
    }
    for ( auto & frag : largeTriFragments )
        if ( frag.f )
            res.set( frag.f );

    return res;
}

void appendGPUVisibleFaces( const Viewport& viewport, const BitSet& pixBs, 
    const std::vector<std::shared_ptr<ObjectMesh>>& objects, 
    std::vector<FaceBitSet>& visibleFaces, bool includeBackfaces /*= true */ )
{
    const auto orthoBackwards = viewport.getBackwardDirection();
    auto gpuPickerVisibleFaces = viewport.findVisibleFaces( pixBs );
    for ( int i = 0; i < objects.size(); ++i )
    {
        const auto& selMesh = objects[i];
        auto it = gpuPickerVisibleFaces.find( selMesh );
        if ( it == gpuPickerVisibleFaces.end() )
            continue;
        if ( !includeBackfaces )
        {
            const auto xf = selMesh->worldXf();
            BitSetParallelFor( it->second, [&] ( FaceId f )
            {
                auto n = selMesh->mesh()->dirDblArea( f );
                Vector3f cameraDir;
                if ( viewport.getParameters().orthographic )
                    cameraDir = orthoBackwards;
                else
                    cameraDir = -viewport.unprojectPixelRay( to2dim( viewport.projectToViewportSpace( selMesh->mesh()->triCenter( f ) ) ) ).d;
                if ( dot( xf.A * n, cameraDir ) < 0 )
                    it->second.set( f, false );
            } );
        }
        visibleFaces[i] |= it->second;
    }
}

VertBitSet findVertsInViewportArea( const Viewport& viewport, const BitSet& pixBs, const ObjectPoints& obj,
                                    bool includeBackfaces /*= true */, bool onlyVisible /*= false */ )
{
    if ( pixBs.none() )
        return {};

    const auto& pointCloud = obj.pointCloud();
    const auto& vpRect = viewport.getViewportRect();
    const auto xf = obj.worldXf();

    auto toClipSpace = [&]( const Vector3f& objPoint )
    {
        const auto p = xf( objPoint );
        return viewport.projectToClipSpace( p );
    };

    auto width = MR::width( vpRect );
    auto height = MR::height( vpRect );

    auto inSelectedArea = [&]( const Vector3f& clipSpacePoint )
    {
        if ( clipSpacePoint[0] < -1.f || clipSpacePoint[0] > 1.f || clipSpacePoint[1] < -1.f || clipSpacePoint[1] > 1.f )
            return false;
        auto x = std::lround( ( clipSpacePoint[0] / 2.f + 0.5f ) * width );
        auto y = std::lround( ( -clipSpacePoint[1] / 2.f + 0.5f ) * height );
        return pixBs.test( x + y * long( width ) );
    };

    // find all verts inside
    auto verts = pointCloud->validPoints;
    const auto orthoBackwards = viewport.getBackwardDirection();
    const auto& normals = pointCloud->normals;
    const bool excludeBackface = !includeBackfaces && normals.size() >= pointCloud->points.size();
    const auto& clippingPlane = viewport.getParameters().clippingPlane;
    BitSetParallelFor( verts, [&]( VertId i )
    {
        if ( !inSelectedArea( toClipSpace( pointCloud->points[i] ) ) )
            verts.set( i, false );
        else if ( excludeBackface )
        {
            Vector3f cameraDir;
            if ( viewport.getParameters().orthographic )
                cameraDir = orthoBackwards;
            else
                cameraDir = -viewport.unprojectPixelRay( to2dim( viewport.projectToViewportSpace( pointCloud->points[i] ) ) ).d;
            if ( dot( xf.A * normals[i], cameraDir ) < 0 )
                verts.set( i, false );
        }
        if ( onlyVisible && obj.globalClippedByPlane( viewport.id ) &&
            clippingPlane.distance( xf( pointCloud->points[i] ) ) > 0 )
        {
            verts.set( i, false );
        }
    } );

    return verts;
}

}
