#include "MRFixUndercuts.h"
#include "MROffset.h"
#include "MRVDBConversions.h"
#include "MRVDBFloatGrid.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRAffineXf3.h"
#include "MRMesh/MRBox.h"
#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRMeshFillHole.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRTimer.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRRingIterator.h"
#include "MRMesh/MRDistanceMap.h"
#include "MRMesh/MRColor.h"
#include "MRMesh/MRMeshIntersect.h"
#include "MRMesh/MRLine3.h"
#include "MRMesh/MRDirMax.h"
#include "MRMesh/MRIntersectionPrecomputes.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MR2to3.h"
#include "MRMesh/MRMeshSubdivide.h"
#include "MRFloatGrid.h"

namespace MR
{

namespace FixUndercuts
{
constexpr float numVoxels = 1e7f;

static void extendAndFillAllHoles( Mesh& mesh, float bottomExtension, const Vector3f& dir )
{
    MR_TIMER;

    auto minV = findDirMax( -dir, mesh, UseAABBTree::YesIfAlreadyConstructed );
    auto borders = extendAllHoles( mesh, Plane3f::fromDirAndPt( dir, mesh.points[minV] - bottomExtension * dir ) );
    fillHoles( mesh, borders );
}

/// move bottom vertices of given mesh to make object thickness at least (minThickness) in (up) direction;
/// use this function before making signed distance field from the mesh with minThickness=voxelSize
/// to avoid unexpected hole appearance in thin areas
static void makeZThickAtLeast( Mesh & mesh, float minThickness, Vector3f up )
{
    MR_TIMER;
    up = up.normalized();
    const IntersectionPrecomputes<float> iprec( up );
    auto newPoints = mesh.points;
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        if ( dot( mesh.pseudonormal( v ), up ) >= 0 )
            return; // skip top surface vertices
        // find up-intersection within minThickness
        auto isec = rayMeshIntersect( mesh, { mesh.points[v], up }, 0.0f, minThickness, &iprec, true,
            [&]( FaceId f )
            {
                VertId a, b, c;
                mesh.topology.getTriVerts( f, a, b, c );
                if ( v == a || v == b || v == c )
                    return false; // ignore intersections with incident faces of (v)
                return dot( mesh.normal( f ), up ) >= 0;
            } );
        // if such intersection found then move bottom point down
        if ( isec )
            newPoints[v] = isec.proj.point - minThickness * up;
    } );

    mesh.points = std::move( newPoints );
}

bool fixGridByAccessor( FloatGrid& grid, openvdb::FloatGrid::Accessor onAccessor, const openvdb::CoordBBox& dimsBB,
    int zOffset, const ProgressCallback& cb )
{
    MR_TIMER;
    auto valAccessor = grid->getAccessor();
    auto maxZ = dimsBB.max().z() - 1;
    auto minZ = dimsBB.min().z();
    auto zDims = dimsBB.max().z() - dimsBB.min().z() + 1 + zOffset;
    for ( int z = maxZ; z + zOffset > minZ; --z )
    {
        bool minZLayer = z + zOffset - 1 == minZ;
        for ( int y = dimsBB.min().y(); y < dimsBB.max().y(); ++y )
        {
            for ( int x = dimsBB.min().x(); x < dimsBB.max().x(); ++x )
            {
                if ( !onAccessor.isValueOn( { x,y,z } ) )
                    continue;
                onAccessor.setValueOn( { x,y,z - 1 } );
                auto valLow = valAccessor.getValue( { x,y,z - 1 } );
                auto val = valAccessor.getValue( { x,y,z } );
                if ( !minZLayer )
                {
                    if ( val < valLow )
                        valAccessor.setValue( { x,y,z - 1 }, val );
                }
                else
                {
                    const float absLimit = 0.3f; // 30% of voxel size
                    val = std::clamp( val, -absLimit, absLimit );
                    valAccessor.setValue( { x,y,z - 1 }, val );
                }
            }
        }
        if ( z % 4 == 0 && !reportProgress( cb, float( maxZ - z + 1 ) / float( zDims ) ) )
            return false;
    }
    return true;
}

bool fixGrid( FloatGrid& grid, int zOffset, const ProgressCallback& cb )
{
    return fixGridByAccessor( grid, grid->getAccessor(), grid->evalActiveVoxelBoundingBox(), zOffset, cb );
}

bool fixGridFullByPart( FloatGrid& full, const FloatGrid& part, int zOffset, const ProgressCallback& cb )
{
    return fixGridByAccessor( full, part->getAccessor(), part->evalActiveVoxelBoundingBox(), zOffset, cb );
}

Expected<void> fix( Mesh& mesh, const FixParams& params )
{
    MR_TIMER;
    MR_WRITER( mesh );

    float voxelSize = params.voxelSize <= 0 ? suggestVoxelSize( mesh, numVoxels ) : params.voxelSize;
    float bottomExtension = params.bottomExtension <= 0.0f ? 2.0f * voxelSize : params.bottomExtension;

    auto rot = AffineXf3f::linear( Matrix3f::rotation( params.findParameters.upDirection, Vector3f::plusZ() ) );

    int zOffset = 0;
    if ( mesh.topology.isClosed() )
        zOffset = int( bottomExtension / voxelSize );

    FaceBitSet regionCpy;
    FloatGrid regionGrid;
    if ( params.region )
    {
        // add new triangles after hole filling to bitset
        regionCpy = *params.region;
    }

    auto cb = params.smooth ? subprogress( params.cb, 0.0f, 0.5f ) : params.cb;


    if ( !reportProgress( cb, 0.05f ) )
        return unexpectedOperationCanceled();

    mesh.transform( rot );
    Vector3f center;
    float heightZ = 0;
    if ( params.findParameters.wallAngle != 0 )
    {
        // transform projective
        auto meshBox = mesh.computeBoundingBox();
        center = meshBox.center();
        auto halfDiagXY = std::sqrt( to2dim( meshBox.size() ).lengthSq() / 2.0f );
        heightZ = halfDiagXY / std::tan( std::abs( params.findParameters.wallAngle ) );


        // need to subdivide - because this projection is not really linear,
        // so for big planar faces back projection will change the geometry
        SubdivideSettings ss;
        ss.maxEdgeLen = 4 * voxelSize; // guarantee some precision but do not subdivide too much
        ss.maxAngleChangeAfterFlip = PI_F / 36.0f;
        ss.maxEdgeSplits = INT_MAX;
        ss.maxDeviationAfterFlip = voxelSize / 3.0f;
        ss.progressCallback = subprogress( cb, 0.05f, 0.25f );
        if ( params.region )
        {
            ss.onEdgeSplit = [&] ( EdgeId e1, EdgeId e )
            {
                if ( regionCpy.test( mesh.topology.left( e ) ) )
                    regionCpy.autoResizeSet( mesh.topology.left( e1 ) );
                if ( regionCpy.test( mesh.topology.right( e ) ) )
                    regionCpy.autoResizeSet( mesh.topology.right( e1 ) );
            };
        }
        subdivideMesh( mesh, ss );
        if ( !reportProgress( cb, 0.25f ) )
            return unexpectedOperationCanceled();

        auto keepGoing = BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
        {
            auto diff = mesh.points[v] - center;
            if ( params.findParameters.wallAngle > 0 )
                diff.z = -diff.z;
            auto ratio = ( diff.z + heightZ ) / heightZ;
            if ( ratio != 0 )
            {
                mesh.points[v].x = center.x + diff.x / ratio;
                mesh.points[v].y = center.y + diff.y / ratio;
            }
        }, subprogress( cb, 0.25f, 0.3f ) );
        if ( !keepGoing )
            return unexpectedOperationCanceled();
    }

    if ( params.region )
        regionCpy.resize( mesh.topology.faceSize(), false );

    extendAndFillAllHoles( mesh, bottomExtension, Vector3f::plusZ() );
    makeZThickAtLeast( mesh, voxelSize, Vector3f::plusZ() );

    if ( params.region )
    {
        regionCpy.resize( mesh.topology.faceSize(), true );

        // create mesh and unclosed grid
        regionGrid = meshToDistanceField( mesh.cloneRegion( regionCpy ), AffineXf3f(), Vector3f::diagonal( voxelSize ), 3.0f, subprogress( cb, 0.3f, 0.4f ) );
        if ( !reportProgress( cb, 0.4f ) )
            return unexpectedOperationCanceled();
    }

    auto fullGrid = meshToLevelSet( mesh, AffineXf3f(), Vector3f::diagonal( voxelSize ), 3.0f, subprogress( cb, 0.4f, 0.6f ) );
    if ( !reportProgress( cb, 0.6f ) )
        return unexpectedOperationCanceled();

    bool keepGoing = true;
    auto fsb = subprogress( cb, 0.6f, 0.8f );
    if ( params.region )
        keepGoing = fixGridFullByPart( fullGrid, regionGrid, zOffset, fsb ); // fix undercuts if fullGrid by active voxels from partGrids
    else
        keepGoing = fixGrid( fullGrid, zOffset, fsb );

    if ( !keepGoing )
        return unexpectedOperationCanceled();

    if ( params.smooth )
    {
        gaussianFilter( fullGrid, 1, 1, subprogress( params.cb, 0.5f, 0.8f ) );
        if ( !reportProgress( params.cb, 0.8f ) )
            return unexpectedOperationCanceled();
    }

    auto meshRes = gridToMesh( std::move( fullGrid ), GridToMeshSettings{
        .voxelSize = Vector3f::diagonal( voxelSize ),
        .cb = subprogress( params.cb,0.8f,0.9f )
    } );

    if ( !meshRes.has_value() )
        return unexpected( std::move( meshRes.error() ) );

    mesh = std::move( *meshRes );

    if ( params.findParameters.wallAngle != 0 )
    {
        // back transform projective
        keepGoing = BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
        {
            auto diff = mesh.points[v] - center;
            if ( params.findParameters.wallAngle > 0 )
                diff.z = -diff.z;
            auto ratio = ( diff.z + heightZ ) / heightZ;
            if ( ratio != 0 )
            {
                mesh.points[v].x = center.x + diff.x * ratio;
                mesh.points[v].y = center.y + diff.y * ratio;
            }
        }, subprogress( params.cb, 0.9f, 0.95f ) );
        if ( !keepGoing )
            return unexpectedOperationCanceled();
    }
    auto rotInversed = rot.inverse();
    mesh.transform( rotInversed );
    return {};
}

Expected<void> fixUndercuts( Mesh& mesh, const Vector3f& upDirectionMeshSpace, float voxelSize, float bottomExtension )
{
    return fix( mesh, { .findParameters = {.upDirection = upDirectionMeshSpace},.voxelSize = voxelSize,.bottomExtension = bottomExtension } );
}

Expected<void> fixUndercuts( Mesh& mesh, const FaceBitSet& faceBitSet, const Vector3f& upDirectionMeshSpace, float voxelSize, float bottomExtension )
{
    return fix( mesh, { .findParameters = {.upDirection = upDirectionMeshSpace},.voxelSize = voxelSize,.bottomExtension = bottomExtension,.region = &faceBitSet } );
}

UndercutMetric getUndercutAreaMetric( const Mesh& mesh )
{
    return [&] ( const FaceBitSet& faces, const FindParams& )
    {
        return mesh.area( faces );
    };
}

UndercutMetric getUndercutAreaProjectionMetric( const Mesh& mesh )
{
    return [&] ( const FaceBitSet& faces, const FindParams& params )
    {
        if ( params.wallAngle == 0.0f )
            return mesh.projArea( params.upDirection, faces );
        auto rot = AffineXf3f::linear( Matrix3f::rotation( params.upDirection, Vector3f::plusZ() ) );
        auto meshBox = mesh.computeBoundingBox( &rot );
        auto halfDiagXY = std::sqrt( to2dim( meshBox.size() ).lengthSq() / 2.0f );
        auto heightZ = halfDiagXY / std::tan( std::abs( params.wallAngle ) );
        if ( params.wallAngle < 0 )
            heightZ = -heightZ;
        auto cameraPoint = meshBox.center() + params.upDirection.normalized() * heightZ;
        return 0.5 * parallel_deterministic_reduce( tbb::blocked_range( 0_f, FaceId{ mesh.topology.faceSize() }, 1024 ), 0.0, [&] ( const auto& range, double curr )
        {
            for ( FaceId f = range.begin(); f < range.end(); ++f )
            {
                if ( !mesh.topology.hasFace( f ) )
                    continue;
                auto dir = ( mesh.triCenter( f ) - cameraPoint ).normalized();
                curr += std::abs( dot( mesh.dirDblArea( f ), dir ) );
            }
            return curr;
        }, [] ( auto a, auto b ) { return a + b; } );
    };
}

template<typename PrimTag>
void findInternal( const Mesh& mesh, TaggedBitSet<PrimTag>& out, const FindParams& params )
{
    const TaggedBitSet<PrimTag>* valids;
    if constexpr ( std::is_same_v<PrimTag, FaceTag> )
    {
        out.resize( mesh.topology.faceSize() );
        valids = &mesh.topology.getValidFaces();
    }
    else
    {
        static_assert( std::is_same_v<PrimTag, VertTag> );
        out.resize( mesh.topology.vertSize() );
        valids = &mesh.topology.getValidVerts();
    }
    std::function<Vector3f( const Vector3f& pos )> getDir;
    std::function<IntersectionPrecomputes<float>* ( )> getPrec;
    IntersectionPrecomputes<float> prec( params.upDirection );
    if ( params.wallAngle == 0.0f )
    {
        getPrec = [&prec] ()->IntersectionPrecomputes<float>* { return &prec; };
        getDir = [&params] ( const Vector3f& )->Vector3f { return params.upDirection; };
    }
    else
    {
        getPrec = [] ()->IntersectionPrecomputes<float>* { return nullptr; };
        auto rot = AffineXf3f::linear( Matrix3f::rotation( params.upDirection, Vector3f::plusZ() ) );
        auto meshBox = mesh.computeBoundingBox( &rot );
        auto halfDiagXY = std::sqrt( to2dim( meshBox.size() ).lengthSq() / 2.0f );
        auto heightZ = halfDiagXY / std::tan( std::abs( params.wallAngle ) );
        if ( params.wallAngle < 0 )
            heightZ = -heightZ;
        auto cameraPoint = meshBox.center() + params.upDirection.normalized() * heightZ;
        bool needflip = dot( cameraPoint - meshBox.center(), params.upDirection ) < 0.0f;
        getDir = [cameraPoint,needflip] ( const Vector3f& pos )->Vector3f
        {
            auto dir = ( cameraPoint - pos ).normalized();
            if ( needflip )
                dir = -dir;
            return dir;
        };
    }
    BitSetParallelFor( *valids, [&] ( Id<PrimTag> p )
    {
        Vector3f pos;
        if constexpr ( std::is_same_v<PrimTag, FaceTag> )
            pos = mesh.triCenter( p );
        else
            pos = mesh.points[p];
        Line3f line = { pos, getDir( pos ) };
        if ( rayMeshIntersect( mesh, line, 0.0f, FLT_MAX, getPrec(), true, [&] (FaceId f)->bool
        {
            if constexpr ( std::is_same_v<PrimTag, FaceTag> )
                return f != p;
            else
            {
                VertId v[3];
                mesh.topology.getTriVerts( f, v );
                return v[0] != p && v[1] != p && v[2] != p;
            }
        } ) )
        {
            out.set( p );
        }
    } );
}

void findUndercuts( const Mesh& mesh, const Vector3f& upDirection, FaceBitSet& outUndercuts )
{
    find( mesh, { .upDirection = upDirection }, outUndercuts );
}

double findUndercuts( const Mesh& mesh, const Vector3f& upDirection, FaceBitSet& outUndercuts, const UndercutMetric& metric )
{
    return find( mesh, { .upDirection = upDirection }, outUndercuts, metric );
}

void findUndercuts( const Mesh& mesh, const Vector3f& upDirection, VertBitSet& outUndercuts )
{
    find( mesh, { .upDirection = upDirection }, outUndercuts );
}

double find( const Mesh& mesh, const FindParams& params, FaceBitSet& outUndercuts, const UndercutMetric& metric /*= {} */ )
{
    MR_TIMER;
    findInternal( mesh, outUndercuts, params );
    if ( metric )
        return metric( outUndercuts, params );
    return DBL_MAX;
}

void find( const Mesh& mesh, const FindParams& params, VertBitSet& outUndercuts )
{
    MR_TIMER;
    findInternal( mesh, outUndercuts, params );
}

double scoreUndercuts( const Mesh& mesh, const Vector3f& upDirection, const Vector2i& resolution )
{
    MR_TIMER;
    auto dir = upDirection.normalized();
    // find whole mesh projected area
    const double meshProjArea =  mesh.projArea( dir );

    // prepare distance map
    auto perp = dir.perpendicular();

    auto& perp1 = perp.first;
    auto& perp2 = perp.second;

    Matrix3f rotationMat{ perp1,perp2,-dir };

    MeshToDistanceMapParams params( rotationMat, resolution, mesh, true );
    auto dm = computeDistanceMap( mesh, params );

    double pixelArea = std::sqrt( params.xRange.lengthSq() * params.yRange.lengthSq() ) / ( double( resolution.x ) * resolution.y );
    /* debug
    {
        static int counter = 0;
        saveDistanceMapToImage( dm, std::to_string( ++counter ) + ".png" );
    }*/

    // find distance map active pixels area
    tbb::enumerable_thread_specific<double> pixelAreaPerThread( 0.0 );
    ParallelFor( 0, resolution.x * resolution.y, pixelAreaPerThread, [&]( int i, double & local )
    {
        if ( dm.isValid( i ) )
            local += pixelArea;
    } );
    double sumPixelsArea = 0.0;
    for ( const auto& papt : pixelAreaPerThread )
        sumPixelsArea += papt;

    // whole area - distance map area
    return meshProjArea - sumPixelsArea;
}

Vector3f improveDirectionInternal( const Mesh& mesh, const DistMapImproveDirectionParameters& params, const UndercutMetric* metric )
{
    MR_TIMER;
    Vector3f dir = params.hintDirection.normalized();
    FaceBitSet undercuts;

    std::function<double( const Vector3f& candidateDir, FaceBitSet* out )> metricFinder;
    if ( metric )
    {
        metricFinder = [&]( const Vector3f& candidateDir, FaceBitSet* out )->double
        {
            return find( mesh, { .upDirection = candidateDir }, *out, *metric );
        };
    }
    else
    {
        metricFinder = [&]( const Vector3f& candidateDir, FaceBitSet* )->double
        {
            return scoreUndercuts( mesh, candidateDir, params.distanceMapResolution );
        };
    }

    double minMetric = metricFinder( dir, &undercuts );

    auto perp = dir.perpendicular();

    auto& perp1 = perp.first;
    auto& perp2 = perp.second;

    const int baseAngNum = ( params.baseAngleStep != 0.0f && params.maxBaseAngle >= params.baseAngleStep ) ? int( params.maxBaseAngle / params.baseAngleStep ) : 1;
    const int polarAngNum = (params.polarAngleStep != 0.0f && 2.0f * PI_F >= params.polarAngleStep) ? int( 2.0f * PI_F / params.polarAngleStep ) : 1;

    std::vector<double> metrics( size_t( baseAngNum ) * polarAngNum );
    std::vector<Vector3f> dirs( size_t( baseAngNum ) * polarAngNum );
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, metrics.size() ), [&]( const tbb::blocked_range<size_t>& range )
    {
        FaceBitSet undercutsi;
        float angle, polar;
        int polari, basei;
        for ( size_t myPartId = range.begin(); myPartId < range.end(); ++myPartId )
        {
            polari = (int) ( myPartId / baseAngNum );
            basei = (int) ( myPartId % baseAngNum );

            angle = ( basei + 1 ) * params.baseAngleStep;
            polar = polari * params.polarAngleStep;

            auto& newDir = dirs[myPartId];
            newDir = dir * std::cos( angle ) + ( std::sin( polar ) * perp1 + std::cos( polar ) * perp2 ) * std::sin( angle );
            metrics[myPartId] = metricFinder( newDir, &undercutsi );
        }
    } );

    auto minElemIt = std::min_element( metrics.cbegin(), metrics.cend() );
    if ( *minElemIt < minMetric )
    {
        dir = dirs[std::distance( metrics.cbegin(), minElemIt )].normalized();
        minMetric = *minElemIt;
    }
    return dir;
}

Vector3f improveDirection( const Mesh& mesh, const ImproveDirectionParameters& params, const UndercutMetric& metric )
{
    MR_TIMER;
    return improveDirectionInternal( mesh, {params}, &metric );
}

Vector3f distMapImproveDirection( const Mesh& mesh, const DistMapImproveDirectionParameters& params )
{
    MR_TIMER;
    return improveDirectionInternal( mesh, params, nullptr );
}

} // namespace FixUndercuts
} // namespace MR
