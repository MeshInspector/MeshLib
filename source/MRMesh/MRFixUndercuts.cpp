#ifndef MRMESH_NO_VOXEL
#include "MRFixUndercuts.h"
#include "MRMesh.h"
#include "MRMatrix3.h"
#include "MRVector3.h"
#include "MRAffineXf3.h"
#include "MRBox.h"
#include "MRBitSet.h"
#include "MRVDBConversions.h"
#include "MRMeshFillHole.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include "MRPlane3.h"
#include "MRRingIterator.h"
#include "MRDistanceMap.h"
#include "MRColor.h"
#include "MRFloatGrid.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRPch/MRTBB.h"
#include "MRPch/MRSpdlog.h"
#include <filesystem>

namespace MR
{

namespace FixUndercuts
{
constexpr float numVoxels = 1e7f;

FloatGrid setupGridFromMesh( Mesh& mesh, const AffineXf3f& rot, float voxelSize, float holeExtension, Vector3f dir )
{
    MR_TIMER;
    auto borders = mesh.topology.findHoleRepresentiveEdges();
    
    for ( auto& border : borders )
        border = buildBottom( mesh, border, dir, holeExtension );
    FillHoleParams params;
    for ( const auto& border : borders )
        fillHole( mesh, border, params );

    return meshToLevelSet( mesh, rot, Vector3f::diagonal( voxelSize ) );
}

void fix( FloatGrid& grid, int zOffset )
{
    MR_TIMER;
    auto dimsBB = grid->evalActiveVoxelBoundingBox();
    auto accessor = grid->getAccessor();

    for ( int z = dimsBB.max().z() - 1; z + zOffset > dimsBB.min().z(); --z )
    {
        for ( int y = dimsBB.min().y(); y < dimsBB.max().y(); ++y )
        {
            for ( int x = dimsBB.min().x(); x < dimsBB.max().x(); ++x )
            {
                if ( !accessor.isValueOn( {x,y,z} ) )
                    continue;
                accessor.setValueOn( {x,y,z - 1} );

                auto valLow = accessor.getValue( {x,y,z - 1} );
                auto val = accessor.getValue( {x,y,z} );
                if ( val < valLow )
                    accessor.setValue( {x,y,z - 1}, val );
            }
        }
    }
}

void fixFullByPart( FloatGrid& full, const FloatGrid& part, int zOffset )
{
    MR_TIMER;
    auto dimsBB = part->evalActiveVoxelBoundingBox();
    auto partAccessor = part->getAccessor();
    auto fullAccessor = full->getAccessor();
    for ( int z = dimsBB.max().z() - 1; z + zOffset > dimsBB.min().z(); --z )
    {
        for ( int y = dimsBB.min().y(); y < dimsBB.max().y(); ++y )
        {
            for ( int x = dimsBB.min().x(); x < dimsBB.max().x(); ++x )
            {
                if ( !partAccessor.isValueOn( {x,y,z} ) )
                    continue;
                partAccessor.setValueOn( {x,y,z - 1} );
                auto valLow = fullAccessor.getValue( {x,y,z - 1} );
                auto val = fullAccessor.getValue( {x,y,z} );
                if ( val < valLow )
                    fullAccessor.setValue( {x,y,z - 1}, val );
            }
        }
    }
}

void fixUndercuts( Mesh& mesh, const Vector3f& upDirectionMeshSpace, float voxelSize, float bottomExtension )
{
    MR_TIMER;
    MR_WRITER( mesh );
    if ( voxelSize == 0.0f )
    {
        // count voxel size if needed
        auto bbox = mesh.computeBoundingBox();
        auto volume = bbox.volume();
        voxelSize = std::cbrtf( volume / numVoxels );
    }
    if ( bottomExtension <= 0.0f )
        bottomExtension = 2.0f * voxelSize;

    auto rot = AffineXf3f::linear( Matrix3f::rotation( upDirectionMeshSpace, Vector3f::plusZ() ) );

    int zOffset = 0;
    if ( mesh.topology.isClosed() )
        zOffset = int( bottomExtension / voxelSize );

    auto grid = setupGridFromMesh( mesh, rot, voxelSize, bottomExtension, upDirectionMeshSpace );
    fix( grid, zOffset );
    
    mesh = gridToMesh( grid, Vector3f::diagonal( voxelSize ) ).value(); // no callback so cannot be stopped
    auto rotInversed = rot.inverse();
    mesh.transform( rotInversed );
}

void fixUndercuts( Mesh& mesh, const FaceBitSet& faceBitSet, const Vector3f& upDirectionMeshSpace, float voxelSize, float bottomExtension )
{
    MR_TIMER;
    MR_WRITER( mesh );
    if ( voxelSize == 0.0f )
    {
        // count voxel size if needed
        auto bbox = mesh.computeBoundingBox();
        auto volume = bbox.volume();
        voxelSize = std::cbrtf( volume / numVoxels );
    }
    if ( bottomExtension <= 0.0f )
        bottomExtension = 2.0f * voxelSize;

    auto rot = AffineXf3f::linear( Matrix3f::rotation( upDirectionMeshSpace, Vector3f::plusZ() ) );

    int zOffset = 0;
    if ( mesh.topology.isClosed() )
        zOffset = int( bottomExtension / voxelSize );

    // add new triangles after hole filling to bitset
    FaceBitSet copyFBS = faceBitSet;
    copyFBS.resize( mesh.topology.faceSize(), false );
    auto fullGrid = setupGridFromMesh( mesh, rot, voxelSize, bottomExtension, upDirectionMeshSpace );
    copyFBS.resize( mesh.topology.faceSize(), true );

    // create mesh and unclosed grid 
    Mesh selectedPartMesh;
    selectedPartMesh.addPartByMask( mesh, copyFBS );
    auto partGrid = meshToDistanceField( selectedPartMesh, rot, Vector3f::diagonal( voxelSize ) );

    // fix undercuts if fullGrid by active voxels from partGrid
    fixFullByPart( fullGrid, partGrid, zOffset );

    // create mesh and restore transform
    mesh = gridToMesh( fullGrid, Vector3f::diagonal( voxelSize ) ).value(); // no callback so cannot be stopped
    auto rotInversed = rot.inverse();
    mesh.transform( rotInversed );
}

UndercutMetric getUndercutAreaMetric( const Mesh& mesh )
{
    return [&]( const FaceBitSet& faces, const Vector3f& )
    {
        return mesh.area( faces );
    };
}

UndercutMetric getUndercutAreaProjectionMetric( const Mesh& mesh )
{
    return [&]( const FaceBitSet& faces, const Vector3f& upDir )
    {
        Vector3f dir = upDir.normalized();
        double twiceRes = 0;
        for ( const auto& f : faces )
        {
            twiceRes += std::abs( dot( mesh.dirDblArea( f ), dir ) );
        }
        return twiceRes * 0.5;
    };
}

double findUndercuts( const Mesh& mesh, const Vector3f& upDirection, FaceBitSet& outUndercuts, const UndercutMetric& metric )
{
    MR_TIMER;

    outUndercuts.resize( mesh.topology.faceSize() );
    float moveUpRay = mesh.computeBoundingBox().diagonal() * 1e-5f; //to be independent of mesh size

    BitSetParallelFor( mesh.topology.getValidFaces(), [&]( FaceId f )
    {
        auto center = mesh.triCenter( f );
        if ( rayMeshIntersect( mesh, { center, upDirection }, moveUpRay ) )
        {
            outUndercuts.set( f );
        }
    } );
    return metric( outUndercuts, upDirection );
}

void findUndercuts( const Mesh& mesh, const Vector3f& upDirection, VertBitSet& outUndercuts )
{
    MR_TIMER;

    outUndercuts.resize( mesh.topology.vertSize() );
    float moveUpRay = mesh.computeBoundingBox().diagonal() * 1e-5f; //to be independent of mesh size
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        if ( rayMeshIntersect( mesh, { mesh.points[v], upDirection }, moveUpRay ) )
        {
            outUndercuts.set( v );
        }
    } );
}

double scoreUndercuts( const Mesh& mesh, const Vector3f& upDirection, const Vector2i& resolution )
{
    MR_TIMER;
    auto dir = upDirection.normalized();
    // find whole mesh projected area
    tbb::enumerable_thread_specific<double> areaPerThread( 0.0 );
    BitSetParallelFor( mesh.topology.getValidFaces(), [&]( FaceId id )
    {
        areaPerThread.local() += ( std::abs( dot( mesh.dirDblArea( id ), dir ) ) * 0.5 );
    } );
    double meshProjArea = 0.0;
    for ( const auto& apt : areaPerThread )
        meshProjArea += apt;

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
    tbb::parallel_for( tbb::blocked_range<int>( 0, resolution.x * resolution.y ),
        [&]( const tbb::blocked_range<int>& range )
    {
        auto& local = pixelAreaPerThread.local();
        for ( int i = range.begin(); i < range.end(); ++i )
        {
            if ( dm.isValid( i ) )
                local += pixelArea;
        }
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
            return findUndercuts( mesh, candidateDir, *out, *metric );
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

    int baseAngNum = int( params.maxBaseAngle / params.baseAngleStep );
    int polarAngNum = int( 2.0f * PI_F / params.polarAngleStep );

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
#endif
