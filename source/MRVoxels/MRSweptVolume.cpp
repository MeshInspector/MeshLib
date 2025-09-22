#include "MRSweptVolume.h"

#include "MRMesh/MRAABBTreePolyline.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRConvexHull.h"
#include "MRMesh/MREndMill.h"
#include "MRMesh/MRExtractIsolines.h"
#include "MRMesh/MRInplaceStack.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMovementBuildBody.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRPolyline2Intersect.h"
#include "MRMesh/MRPolylineProject.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRTriMesh.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRPch/MRFmt.h"
#include "MRVoxels/MRFloatGrid.h"
#include "MRVoxels/MRMarchingCubes.h"
#include "MRVoxels/MRVDBConversions.h"
#include "MRVoxels/MRVDBFloatGrid.h"

namespace
{

using namespace MR;

bool fuzzyEqual( const Vector3f& a, const Vector3f& b, float eps )
{
    return ( a - b ).lengthSq() <= sqr( eps );
}

// https://stackoverflow.com/a/28139075
template <typename T>
struct reversion_wrapper { T& iterable; };
template <typename T>
auto begin( reversion_wrapper<T> w ) { return std::rbegin( w.iterable ); }
template <typename T>
auto end( reversion_wrapper<T> w ) { return std::rend( w.iterable ); }
template <typename T>
reversion_wrapper<T> reversed( T&& iterable ) { return { iterable }; }

Contour2f makeToolOutline( const MeshPart& tool )
{
    auto vertices = tool.mesh.topology.getValidVerts();
    if ( tool.region )
    {
        BitSetParallelFor( vertices, [&] ( VertId v )
        {
            if ( !tool.mesh.topology.isInnerOrBdVertex( v, tool.region ) )
                vertices.set( v, false );
        } );
    }

    std::vector<Vector2f> points;
    points.reserve( vertices.count() );
    for ( const auto v : vertices )
    {
        const auto& p = tool.mesh.points[v];
        points.emplace_back( Vector2f{ p.x, p.y }.length(), p.z );
    }

    auto outline = makeConvexHull( std::move( points ) );

    // close the contour
    for ( const auto& p : reversed( outline ) )
        if ( p.x >= 1e-6f )
            outline.emplace_back( -p.x, +p.y );
    if ( outline.back() != outline.front() )
        outline.emplace_back( outline.front() );

    return outline;
}

Contours3f makeToolProfile( const MeshPart& tool )
{
    Contours3f results;
    for ( const auto& section : extractPlaneSections( tool, { Vector3f::minusX(), 0.f } ) )
    {
        Contour3f contour;
        contour.reserve( section.size() );

        Vector3f prevP;
        for ( const auto& ep : section )
        {
            const auto p = tool.mesh.edgePoint( ep );
            if ( contour.empty() || !fuzzyEqual( p, prevP, 1e-6f ) )
                contour.emplace_back( p );
            prevP = p;
        }

        results.emplace_back( std::move( contour ) );
    }
    return results;
}

float getBoundarySignedDistanceSq( const Box2f& box, const Vector2f& p )
{
    float posSum = 0.f;
    float maxNeg = -FLT_MAX;
    for ( auto i = 0; i < box.elements; ++i )
    {
        const auto dist = std::max( box.min[i] - p[i], p[i] - box.max[i] );
        if ( dist > 0.f )
            posSum += sqr( dist );
        else
            maxNeg = std::max( maxNeg, dist );
    }
    return posSum > 0.f ? posSum : -sqr( maxNeg );
}

float sqrSgn( float v )
{
    return (float)sgn( v ) * sqr( v );
}

float sqrtSgn( float v )
{
    return (float)sgn( v ) * std::sqrt( std::abs( v ) );
}

float getDistanceSq( const LineSegm2f& segm, const Vector2f& p )
{
    return ( closestPointOnLineSegm( p, segm ) - p ).lengthSq();
}

bool ccw( const LineSegm2f& segm, const Vector2f& p )
{
    return cross( segm.b - segm.a, p - segm.a ) > 0.f;
}

template <typename Callback>
void findEdgesInTool( const Polyline3& polyline, const Vector3f& pos, float toolRadius, float toolMinHeight, float toolMaxHeight, Callback&& cb )
{
    const auto& tree = polyline.getAABBTree();
    if ( tree.nodes().empty() )
        return;

    const auto pos2 = Vector2f{ pos };
    const auto toolRadiusSq = sqr( toolRadius );

    Box3f toolBox {
        { -toolRadius, -toolRadius, -toolMaxHeight },
        { +toolRadius, +toolRadius, -toolMinHeight },
    };
    toolBox.min += pos;
    toolBox.max += pos;

    InplaceStack<NoInitNodeId, 32> subtasks;

    auto addSubTask = [&] ( NodeId n )
    {
        const auto& box = tree.nodes()[n].box;
        if ( box.intersection( toolBox ).valid() )
            subtasks.push( n );
    };

    addSubTask( tree.rootNodeId() );

    while ( !subtasks.empty() )
    {
        const auto n = subtasks.top();
        subtasks.pop();
        const auto& node = tree[n];

        if ( node.leaf() )
        {
            const auto segm = polyline.edgeSegment( node.leafId() );
            const auto proj = closestPointOnLineSegm( pos, segm );

            const auto dist2Sq = ( Vector2f{ proj } - pos2 ).lengthSq();
            if ( toolRadiusSq < dist2Sq )
                continue;

            const Vector2f toolPos { std::sqrt( dist2Sq ), pos.z - proj.z };
            if ( std::forward<Callback>( cb )( toolPos ) )
                return;
        }
        else
        {
            addSubTask( node.r ); // look at right node later
            addSubTask( node.l ); // look at left node first
        }
    }
}

constexpr int cVoxelPadding = 3;

}

namespace MR
{

Box3f computeWorkArea( const Polyline3& toolpath, const Box2f& toolBox2 )
{
    auto workArea = toolpath.computeBoundingBox();

    const auto radius = toolBox2.max.x;
    const Box3f toolBox {
        { -radius, -radius, toolBox2.min.y },
        { +radius, +radius, toolBox2.max.y },
    };
    workArea.min += toolBox.min;
    workArea.max += toolBox.max;

    return workArea;
}

Box3f computeWorkArea( const Polyline3& toolpath, const MeshPart& tool )
{
    auto workArea = toolpath.computeBoundingBox();

    const auto toolBox = tool.mesh.computeBoundingBox( tool.region );
    workArea.min += toolBox.min;
    workArea.max += toolBox.max;

    return workArea;
}

Box3i computeGridBox( const Box3f& workArea, float voxelSize )
{
    Box3f gridBox { workArea.min / voxelSize, workArea.max / voxelSize };
    for ( auto dim = 0; dim < gridBox.elements; ++dim )
    {
        gridBox.min[dim] = std::floor( gridBox.min[dim] );
        gridBox.max[dim] = std::ceil( gridBox.max[dim] );
    }

    return Box3i{ gridBox }.expanded( Vector3i::diagonal( cVoxelPadding ) );
}

Expected<Mesh> computeSweptVolumeWithMeshMovement( const ComputeSweptVolumeParameters& params )
{
    FloatGrid grid = std::make_shared<OpenVdbFloatGrid>();
    setLevelSetType( grid );
    openvdb::tools::changeBackground( grid->tree(), 9999.f );

    const auto toolProfile = makeToolProfile( params.toolMesh );

    const auto tool = params.toolMesh.region ? params.toolMesh.mesh.cloneRegion( *params.toolMesh.region ) : params.toolMesh.mesh;

    if ( !reportProgress( params.cb, 0.01f ) )
        return unexpectedOperationCanceled();

    const auto toolpaths = params.path.contours();

    auto cb1 = subprogress( params.cb, 0.01f, 0.60f );
    auto i = 0;

    for ( const auto& path : toolpaths )
    {
        auto vol = makeMovementBuildBody( toolProfile, { path }, {
            .allowRotation = true,
            .center = Vector3f { 0.f, 0.f, 0.f },
        } );

        auto end = tool;
        end.transform( AffineXf3f::translation( path.front() ) );
        vol.addMesh( end );
        if ( path.back() != path.front() )
        {
            end = tool;
            end.transform( AffineXf3f::translation( path.back() ) );
            vol.addMesh( end );
        }

        auto cbi = subprogress( cb1, i++, toolpaths.size() );
        auto ls = meshToLevelSet( vol, {}, Vector3f::diagonal( params.voxelSize ), 3.f, cbi );
        if ( !ls )
            return unexpectedOperationCanceled();
        grid += ls;
    }

    auto mesh = gridToMesh( grid, {
        .voxelSize = Vector3f::diagonal( params.voxelSize ),
        .cb = subprogress( params.cb, 0.60f, 1.00f ),
    } );
    if ( !mesh )
        return unexpected( mesh.error() == stringOperationCanceled() ? mesh.error() : fmt::format( "Failed to build mesh: {}", mesh.error() ) );

    return mesh;
}

Expected<Mesh> computeSweptVolumeWithDistanceVolume( const ComputeSweptVolumeParameters& params, const Box2f& toolBox, auto&& posToDistFunc )
{
    MR_TIMER;

    const auto workArea = computeWorkArea( params.path, toolBox );
    const auto gridBox = computeGridBox( workArea, params.voxelSize );
    const auto dims = gridBox.size();
    const auto origin = Vector3f{ gridBox.min } * params.voxelSize;
    const auto padding = (float)cVoxelPadding * params.voxelSize;
    const auto paddingSq = sqr( padding );

    const auto getter = [&] ( const Vector3i& vox ) -> float
    {
        const auto pos = ( Vector3f{ vox } + Vector3f::diagonal( 0.5f ) ) * params.voxelSize + origin;

        auto distSgnSq = FLT_MAX;
        findEdgesInTool( params.path, pos, toolBox.max.x + padding, toolBox.min.y - padding, toolBox.max.y + padding, [&] ( const Vector2f& toolPos )
        {
            distSgnSq = std::min( distSgnSq, std::forward<decltype( posToDistFunc )>( posToDistFunc )( toolPos ) );
            return distSgnSq < -paddingSq;
        } );
        return distSgnSq != FLT_MAX ? sqrtSgn( distSgnSq ) : FLT_MAX;
    };

    VolumeIndexer indexer( dims );
    SimpleVolume volume {
        .voxelSize = Vector3f::diagonal( params.voxelSize ),
    };
    size_t maxSliceCount = dims.z;
    if ( params.memoryLimit )
    {
        const auto sliceSize = indexer.sizeXY() * sizeof( float );
        maxSliceCount = std::min( params.memoryLimit / sliceSize, maxSliceCount );
        constexpr size_t cMinSliceCount = 2;
        if ( maxSliceCount < cMinSliceCount )
            return unexpected( "Insufficient memory limits" );
    }
    volume.data.resize( indexer.sizeXY() * maxSliceCount );

    Timer timer( "" );
    MarchingCubesByParts mesher( dims, {
        .origin = origin,
        .cb = subprogress( params.cb, 0.00f, 0.90f ),
        .iso = 0.f,
        .lessInside = true,
        .freeVolume = [&] { decltype( volume.data )().swap( volume.data ); }
    } );
    for ( auto begin = 0; begin + 1 < dims.z; begin = mesher.nextZ() )
    {
        const auto end = std::min( begin + (int)maxSliceCount, dims.z );
        volume.dims = { dims.x, dims.y, end - begin };

        timer.restart( "compute distance volume" );
        const auto offset = indexer.sizeXY() * begin;
        ParallelFor( indexer.toVoxelId( { 0, 0, begin } ), indexer.toVoxelId( { 0, 0, end } ), [&] ( VoxelId vox )
        {
            volume.data.vec_[vox.get() - offset] = getter( indexer.toPos( vox ) );
        } );

        timer.restart( "triangulating distance volume" );
        if ( auto res = mesher.addPart( volume ); !res )
            return unexpected( std::move( res.error() ) );
    }
    timer.restart( "building mesh" );
    return mesher.finalize().transform( [&] ( TriMesh&& mesh )
    {
        return Mesh::fromTriMesh( std::move( mesh ), {}, subprogress( params.cb, 0.90f, 1.00f ) );
    } );
}

Expected<Mesh> computeSweptVolumeWithDistanceVolume( const ComputeSweptVolumeParameters& params )
{
    if ( params.toolSpec )
    {
        const auto& toolSpec = *params.toolSpec;
        const auto radius = toolSpec.diameter / 2.f;
        const Box2f toolBox {
            { -radius, 0.f },
            { +radius, toolSpec.length },
        };

        using Cutter = EndMillCutter::Type;
        switch ( toolSpec.cutter.type )
        {
        case Cutter::Flat:
            return computeSweptVolumeWithDistanceVolume( params, toolBox, [toolBox] ( const Vector2f& toolPos )
            {
                return getBoundarySignedDistanceSq( toolBox, toolPos );
            } );

        case Cutter::Ball:
        {
            const Vector2f center { 0.f, radius };
            return computeSweptVolumeWithDistanceVolume( params, toolBox, [radius, toolBox, center] ( const Vector2f& toolPos )
            {
                if ( toolPos.y <= center.y )
                    return sqrSgn( ( toolPos - center ).length() - radius );
                else
                    return getBoundarySignedDistanceSq( toolBox, toolPos );
            } );
        }

        case Cutter::BullNose:
        {
            const auto cornerRadius = toolSpec.cutter.cornerRadius;
            const Vector2f center { radius - cornerRadius, cornerRadius };
            return computeSweptVolumeWithDistanceVolume( params, toolBox, [cornerRadius, toolBox, center] ( const Vector2f& toolPos )
            {
                if ( center.x <= toolPos.x && toolPos.y <= center.y )
                    return sqrSgn( ( toolPos - center ).length() - cornerRadius );
                else
                    return getBoundarySignedDistanceSq( toolBox, toolPos );
            } );
        }

        case Cutter::Chamfer:
        {
            const auto endRadius = toolSpec.cutter.endDiameter / 2.f;
            const auto cutterHeight = toolSpec.getMinimalCutLength();
            const LineSegm2f slope {
                { endRadius, 0.f },
                { radius, cutterHeight },
            };
            return computeSweptVolumeWithDistanceVolume( params, toolBox, [endRadius, cutterHeight, slope, toolBox] ( const Vector2f& toolPos )
            {
                if ( endRadius <= toolPos.x && toolPos.y <= cutterHeight )
                    return getDistanceSq( slope, toolPos ) * ( ccw( slope, toolPos ) ? -1.f : +1.f );
                else
                    return getBoundarySignedDistanceSq( toolBox, toolPos );
            } );
        }

        case Cutter::Count:
            MR_UNREACHABLE
        }
        MR_UNREACHABLE
    }
    else
    {
        const auto outline = makeToolOutline( params.toolMesh );
        // TODO: optimize
        Polyline2 toolPolyline( outline );
        const auto toolBox = toolPolyline.getBoundingBox();

        return computeSweptVolumeWithDistanceVolume( params, toolBox, [&] ( const Vector2f& toolPos )
        {
            const auto toolDistSq = findProjectionOnPolyline2( toolPos, toolPolyline ).distSq;
            const auto toolDistSgn = isPointInsidePolyline( toolPolyline, toolPos ) ? -1.f : +1.f;
            return toolDistSq * toolDistSgn;
        } );
    }
}

Expected<Mesh> computeSweptVolumeWithCustomToolDistance( IComputeToolDistance& comp, const ComputeSweptVolumeParameters& params )
{
    const auto outline = makeToolOutline( params.toolMesh );
    // TODO: optimize
    Polyline2 toolProfile( outline );

    const auto workArea = computeWorkArea( params.path, params.toolMesh );
    const auto gridBox = computeGridBox( workArea, params.voxelSize );
    const auto dims = gridBox.size();
    const auto origin = Vector3f{ gridBox.min } * params.voxelSize;
    const auto padding = (float)cVoxelPadding * params.voxelSize;

    Expected<Vector3i> maxDims;
    if ( params.toolSpec )
        maxDims = comp.prepare( dims, params.path, *params.toolSpec );
    else
        maxDims = comp.prepare( dims, params.path, toolProfile );
    MR_RETURN_IF_UNEXPECTED( maxDims )
    if ( maxDims->x < dims.x || maxDims->y < dims.y )
        return unexpected( "Incompatible dims" );
    const size_t maxSliceCount = maxDims->z;

    SimpleVolume volume {
        .dims = dims,
        .voxelSize = Vector3f::diagonal( params.voxelSize ),
    };
    VolumeIndexer indexer( dims );
    volume.data.resize( indexer.sizeXY() * maxSliceCount );
    if ( auto res = comp.computeToolDistance( volume.data, volume.dims, params.voxelSize, origin, padding ); !res )
        return unexpected( std::move( res.error() ) );

    Timer timer( "" );
    MarchingCubesByParts mesher( dims, {
        .origin = origin,
        .cb = subprogress( params.cb, 0.00f, 0.90f ),
        .iso = 0.f,
        .lessInside = true,
        .freeVolume = [&] { decltype( volume.data )().swap( volume.data ); }
    } );
    for ( auto begin = 0; begin + 1 < dims.z; begin = mesher.nextZ() )
    {
        const auto end = std::min( begin + (int)maxSliceCount, dims.z );
        volume.dims = { dims.x, dims.y, end - begin };

        timer.restart( "compute distance volume" );
        const auto shift = Vector3f{ 0, 0, (float)begin } * params.voxelSize;
        // TODO: async
        MR_RETURN_IF_UNEXPECTED( comp.computeToolDistance( volume.data, volume.dims, params.voxelSize, origin + shift, padding ) )

        timer.restart( "triangulating distance volume" );
        MR_RETURN_IF_UNEXPECTED( mesher.addPart( volume ) )
    }
    timer.restart( "building mesh" );
    return mesher.finalize().transform( [&] ( TriMesh&& mesh )
    {
        return Mesh::fromTriMesh( std::move( mesh ), {}, subprogress( params.cb, 0.90f, 1.00f ) );
    } );
}

} // namespace MR
