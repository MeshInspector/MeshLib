#ifndef MESHLIB_NO_VOXELS

#include "MRVoxels/MRVoxelsConversionsByParts.h"
#include "MRVoxels/MRVDBFloatGrid.h"
#include "MRVoxels/MRVoxelsVolume.h"
#include "MRVoxels/MRMarchingCubes.h"
#include "MRMesh/MRGTest.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRTriMesh.h"

namespace MR
{

TEST( MRMesh, volumeToMeshByParts )
{
    const Vector3i dimensions { 101, 101, 101 };
    constexpr float radius = 50.f;
    constexpr Vector3f center { 50.f, 50.f, 50.f };

    constexpr float voxelSize = 0.01f;

    VolumePartBuilder<VdbVolume> vdbBuilder = [&] ( int begin, int end, std::optional<Vector3i>& )
    {
        auto grid = MakeFloatGrid( openvdb::FloatGrid::create() );
        grid->setGridClass( openvdb::GRID_LEVEL_SET );

        auto accessor = grid->getAccessor();
        for ( auto z = 0; z < dimensions.z; ++z )
        {
            for ( auto y = 0; y < dimensions.y; ++y )
            {
                for ( auto x = begin; x < end; ++x )
                {
                    const Vector3f pos( (float)x, (float)y, (float)z );
                    const auto dist = ( center - pos ).length();
                    accessor.setValue( { x, y, z }, dist - radius );
                }
            }
        }

        return VdbVolume {
            {
                .data = std::move( grid ),
                .dims = { end - begin, dimensions.y, dimensions.z },
                .voxelSize = Vector3f::diagonal( voxelSize )
            },
            {
                -radius, //min
                +radius  //max
            }
        };
    };

    VolumePartBuilder<SimpleVolumeMinMax> simpleBuilder = [&] ( int begin, int end, std::optional<Vector3i>& offset )
    {
        SimpleVolumeMinMax result {
            {
                .dims = { end - begin, dimensions.y, dimensions.z },
                .voxelSize = Vector3f::diagonal( voxelSize )
            },
            {
                -radius, //min
                +radius  //max
            }
        };

        VolumeIndexer indexer( result.dims );
        result.data.resize( indexer.size() );

        tbb::parallel_for( tbb::blocked_range<int>( 0, dimensions.z ), [&] ( const tbb::blocked_range<int>& range )
        {
            for ( auto z = range.begin(); z < range.end(); ++z )
            {
                for ( auto y = 0; y < dimensions.y; ++y )
                {
                    for ( auto x = begin; x < end; ++x )
                    {
                        const Vector3f pos( (float)x, (float)y, (float)z );
                        const auto dist = ( center - pos ).length();
                        result.data[indexer.toVoxelId( { x - begin, y, z } )] = dist - radius;
                    }
                }
            }
        } );

        offset = { begin, 0, 0 };

        return result;
    };

    VolumePartBuilder<FunctionVolume> functionBuilder = [&] ( int begin, int end, std::optional<Vector3i>& offset )
    {
        FunctionVolume result {
            .dims = { end - begin, dimensions.y, dimensions.z },
            .voxelSize = Vector3f::diagonal( voxelSize )
        };

        result.data = [radius = radius, offsetCenter = center - Vector3f( (float)begin, 0.f, 0.f )] ( const Vector3i& pos )
        {
            const auto dist = ( offsetCenter - Vector3f( pos ) ).length();
            return dist - radius;
        };

        offset = { begin, 0, 0 };

        return result;
    };

    constexpr size_t memoryUsage = 2 * ( 1 << 20 ); // 2 MiB

    auto vdbMesh = volumeToMeshByParts( vdbBuilder, dimensions, Vector3f::diagonal( voxelSize ), {
        .maxVolumePartMemoryUsage = memoryUsage,
    } );
    EXPECT_TRUE( vdbMesh.has_value() );

    auto simpleMesh = volumeToMeshByParts( simpleBuilder, dimensions, Vector3f::diagonal( voxelSize ), {
        .maxVolumePartMemoryUsage = memoryUsage,
    } );
    EXPECT_TRUE( simpleMesh.has_value() );

    auto functionMesh = volumeToMeshByParts( functionBuilder, dimensions, Vector3f::diagonal( voxelSize ), {
        .maxVolumePartMemoryUsage = memoryUsage,
    } );
    EXPECT_TRUE( functionMesh.has_value() );

    constexpr auto r = radius * voxelSize;
    constexpr auto expectedVolume = 4.f * PI_F * r * r * r / 3.f;
    EXPECT_NEAR( expectedVolume, vdbMesh->volume(), 0.001f );
    EXPECT_NEAR( expectedVolume, simpleMesh->volume(), 0.001f );
    EXPECT_NEAR( expectedVolume, functionMesh->volume(), 0.001f );

    MarchingCubesByParts mc( dimensions, { .iso = 0.f, .lessInside = true } );
    constexpr int zLayersInPart = 2;
    SimpleVolume part
    {
        .dims = { dimensions.x, dimensions.y, zLayersInPart },
        .voxelSize = Vector3f::diagonal( voxelSize )
    };
    part.data.resize( zLayersInPart * size_t( dimensions.x ) * dimensions.y );
    for ( int iz = 0; iz + zLayersInPart <= dimensions.z; ++iz )
    {
        ParallelFor( 0, zLayersInPart, [&]( int l )
        {
            VoxelId i( l * size_t( dimensions.x ) * dimensions.y );
            for ( auto y = 0; y < dimensions.y; ++y )
            {
                for ( auto x = 0; x < dimensions.x; ++x, ++i )
                {
                    const Vector3f pos( (float)x, (float)y, (float)( iz + l ) );
                    const auto dist = ( center - pos ).length();
                    part.data[i] = dist - radius;
                }
            }
        } );
        auto e = mc.addPart( part );
        EXPECT_TRUE( e.has_value() );
    }
    Mesh mesh = Mesh::fromTriMesh( *mc.finalize() );
    EXPECT_NEAR( expectedVolume, mesh.volume(), 0.001f );
}

} //namespace MR

#endif //!MESHLIB_NO_VOXELS
