#include "MRVoxelsConversions.h"
#include "MRMesh.h"
#include "MRVolumeIndexer.h"
#include "MRIntersectionPrecomputes.h"
#include "MRMeshIntersect.h"
#include "MRLine3.h"
#include "MRMeshBuilder.h"
#include "MRPch/MRTBB.h"
#include "MRTimer.h"

namespace MR
{

SimpleVolume meshToSimpleVolume( const Mesh& mesh, const MeshToSimpleVolumeParams& params /*= {} */ )
{
    MR_TIMER
    SimpleVolume res;
    auto transposedBasis = params.basis.A.transposed();
    for ( int i = 0; i < 3; ++i )
        res.voxelSize[i] = transposedBasis[i].length();
    res.dims = params.dimensions;
    VolumeIndexer indexer( res.dims );
    res.data.resize( indexer.size() );
    // used in Winding rule mode
    const IntersectionPrecomputes<double> precomputedInter( Vector3d::plusX() );

    tbb::enumerable_thread_specific<std::pair<float, float>> minMax( { FLT_MAX,-FLT_MAX } );
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, indexer.size() ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            auto coord = Vector3f( indexer.toPos( VoxelId( i ) ) ) + Vector3f::diagonal( 0.5f );
            auto voxelCenter = params.basis.b + params.basis.A * coord;
            float dist{ 0.0f };
            if ( params.signMode != MeshToSimpleVolumeParams::SignDetectionMode::TopologyOrientation )
                dist = std::sqrt( findProjection( voxelCenter, mesh, params.maxDistSq ).distSq );
            else
                dist = findSignedDistance( voxelCenter, mesh )->dist;

            bool changeSign = false;
            if ( params.signMode == MeshToSimpleVolumeParams::SignDetectionMode::WindingRule )
            {
                int numInters = 0;
                rayMeshIntersectAll( mesh, Line3d( Vector3d( voxelCenter ), Vector3d::plusX() ),
                    [&numInters] ( const MeshIntersectionResult& ) mutable
                {
                    ++numInters;
                    return true;
                } );
                changeSign = numInters % 2 == 1; // inside
            }
            if ( changeSign )
                dist = -dist;
            auto& localMinMax = minMax.local();
            if ( dist < localMinMax.first )
                localMinMax.first = dist;
            if ( dist > localMinMax.second )
                localMinMax.second = dist;
            res.data[i] = dist;
        }
    } );
    for ( const auto& [min, max] : minMax )
    {
        if ( min < res.min )
            res.min = min;
        if ( max > res.max )
            res.max = max;
    }
    return res;
}

Mesh simpleVolumeToMesh( const SimpleVolume& volume, const SimpleVolumeToMeshParams& params /*= {} */ )
{
    if ( params.iso <= volume.min || params.iso >= volume.max || 
        volume.dims.x <= 0 || volume.dims.y <= 0 || volume.dims.z <= 0 )
        return {};

    MR_TIMER
    auto transposedBasis = params.basis.A.transposed();
    VolumeIndexer indexer( volume.dims );

    enum NeighborDir
    {
        X, Y, Z, Count
    } dir{ X };

    // point between two neighbor voxels
    struct SeparationPoint 
    {
        Vector3f position; // coordinate
        bool low{ false }; // orientation: true means that baseVoxelId has lower value
        VertId vid;
    };

    const std::array<size_t, size_t( NeighborDir::Count )> cDirStep{
        1,
        volume.dims.x,
        indexer.sizeXY()
    };
    auto setupSeparation = [&] ( size_t base, NeighborDir dir )->SeparationPoint
    {
        auto nextId = base + cDirStep[dir];
        if ( nextId >= volume.dims[dir] )
            return {};
        const auto& valueB = volume.data[base];
        const auto& valueD = volume.data[nextId];
        bool bLower = valueB < params.iso;
        bool dLower = valueD < params.iso;

        if ( bLower == dLower )
            return {};

        const float ratio = std::abs( params.iso - bLower ) / std::abs( dLower - bLower );

        SeparationPoint res;
        res.low = bLower < dLower;
        auto bPos = params.basis.b + params.basis.A * ( Vector3f( indexer.toPos( VoxelId( base ) ) ) + Vector3f::diagonal( 0.5f ) );
        auto dPos = params.basis.b + params.basis.A * ( Vector3f( indexer.toPos( VoxelId( nextId ) ) ) + Vector3f::diagonal( 0.5f ) );
        res.position = ( 1.0f - ratio ) * bPos + ratio * dPos;
        res.vid = VertId{ 0 }; // any valid VertId is ok
        return res;
    };

    using SeparationPointSet = std::array<SeparationPoint, size_t( NeighborDir::Count )>;
    using SeparationPointMap = ParallelHashMap<size_t, SeparationPointSet>;
    SeparationPointMap hmap;
    const auto subcnt = hmap.subcnt();
    // find all separate points
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, subcnt, 1 ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        assert( range.begin() + 1 == range.end() );
        for ( size_t i = 0; i + indexer.sizeXY() < volume.data.size(); ++i )
        {
            auto hashval = hmap.hash( i );
            if ( hmap.subidx( hashval ) != range.begin() )
                continue;

            SeparationPointSet set;
            bool atLeastOneOk = false;
            for ( int n = NeighborDir::X; n < NeighborDir::Count; ++n )
            {
                auto separation = setupSeparation( i, NeighborDir( n ) );
                if ( separation.vid )
                {
                    set[n] = std::move( separation );
                    atLeastOneOk = true;
                }
            }
            if ( !atLeastOneOk )
                continue;

            hmap.insert( { i, set } );
        }
    } );

    // build triangulation per thread
    struct ThreadTriangulation
    {
        Triangulation t;
        int zIndexShift{ 0 };
        int startLayer{ 0 };
    };
    auto checkIter = [&] ( const auto& iter, int mode )
    {
        switch ( mode )
        {
        case 0: // base voxel
            return iter != hmap.cend();
        case 1: // x + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return iter->second[NeighborDir::Y].vid.valid() || iter->second[NeighborDir::Z].vid.valid();
        }
        case 2: // y + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return iter->second[NeighborDir::X].vid.valid() || iter->second[NeighborDir::Z].vid.valid();
        }
        case 3: // x + 1, y + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return iter->second[NeighborDir::Z].vid.valid();
        }
        case 4: // z + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return iter->second[NeighborDir::X].vid.valid() || iter->second[NeighborDir::Y].vid.valid();
        }
        case 5: // x + 1, z + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return iter->second[NeighborDir::Y].vid.valid();
        }
        case 6: // y + 1, z + 1 voxel
        {
            if ( iter == hmap.cend() )
                return false;
            return iter->second[NeighborDir::X].vid.valid();
        }
        default:
            return false;
        }
    };
    tbb::enumerable_thread_specific<ThreadTriangulation> triangulationPerThread;
    tbb::parallel_for( tbb::blocked_range<int>( 0, volume.dims.z - 1 ), [&] ( const tbb::blocked_range<int>& range )
    {
        auto& localTriangulation = triangulationPerThread.local();
        localTriangulation.startLayer = range.begin();
        for ( int z = range.begin(); z < range.end(); ++z )
        {
            bool lastLayer = z + 1 == range.end();
            for ( int y = 0; y + 1 < volume.dims.y; ++y )
            {
                for ( int x = 0; x + 1 < volume.dims.x; ++x )
                {
                    const auto mapIterBase = hmap.find( size_t( indexer.toVoxelId( { x,y,z } ) ) );
                    const auto mapIterX = hmap.find( size_t( indexer.toVoxelId( { x + 1,y,z } ) ) );
                    const auto mapIterY = hmap.find( size_t( indexer.toVoxelId( { x,y + 1,z } ) ) );
                    const auto mapIterXY = hmap.find( size_t( indexer.toVoxelId( { x + 1,y + 1,z } ) ) );
                    const auto mapIterZ = hmap.find( size_t( indexer.toVoxelId( { x,y,z + 1 } ) ) );
                    const auto mapIterXZ = hmap.find( size_t( indexer.toVoxelId( { x + 1,y,z + 1 } ) ) );
                    const auto mapIterYZ = hmap.find( size_t( indexer.toVoxelId( { x,y + 1,z + 1 } ) ) );
                    
                    bool checkZ = checkIter( mapIterZ, 4 ); // need to count index shift

                    if ( !checkIter( mapIterBase, 0 ) &&
                        !checkIter( mapIterX, 1 ) &&
                        !checkIter( mapIterY, 2 ) &&
                        !checkIter( mapIterXY, 3 ) &&
                        !checkZ &&
                        !checkIter( mapIterXZ, 5 ) &&
                        !checkIter( mapIterYZ, 6 ) )
                        continue;
                    if ( lastLayer && checkZ )
                    {
                        if ( mapIterZ->second[NeighborDir::X].vid.valid() )
                            ++localTriangulation.zIndexShift;
                        if ( mapIterZ->second[NeighborDir::Y].vid.valid() )
                            ++localTriangulation.zIndexShift;
                    }

                    // find neighbors
                    // triangulate one voxel
                }
            }
        }
    }, tbb::static_partitioner() );
}

}