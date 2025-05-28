#include "MRTeethMaskToDirectionVolume.h"

#include "MRVDBConversions.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRParallelMinMax.h"
#include "MRMeshToDistanceVolume.h"
#include "MRMesh/MRPointsToMeshProjector.h"
#include "MRMesh/MRMesh.h"

#include <MRPch/MRFmt.h>

namespace
{

constexpr float invalidValue = -1.1f; // invalid value for a component of a normalized vector

void forEachInSimpleVolume( auto& vol, auto f )
{
    MR::VolumeIndexer ind( vol.dims );
    for ( int z = 0; z < ind.dims().z; ++z )
    {
        for ( int x = 0; x < ind.dims().x; ++x )
        {
            for ( int y = 0; y < ind.dims().y; ++y )
            {
                const MR::Vector3i pt( x, y, z );
                f( pt, vol.data[ind.toVoxelId( pt )] );
            }
        }
    }
}

}

namespace MR
{

std::optional<DentalId> DentalId::fromFDI( int id )
{
    int t = id % 10;
    int q = id / 10;
    if ( q >= 1 && q <= 4 && t >= 1 && t <= 8 )
    {
        return DentalId( id );
    }
    else
        return std::nullopt;
}

int DentalId::fdi() const
{
    return fdi_;
}


TeethMaskToDirectionVolumeConvertor::TeethMaskToDirectionVolumeConvertor() = default;

const HashMap<int, Box3i>& TeethMaskToDirectionVolumeConvertor::getObjectBounds() const
{
    return presentObjects_;
}

Expected<TeethMaskToDirectionVolumeConvertor>
    TeethMaskToDirectionVolumeConvertor::create( const MR::VdbVolume& volume,
                                                 const std::vector<int>& additionalIds )
{
    auto maybeSimpleVolume = vdbVolumeToSimpleVolume( volume );
    if ( !maybeSimpleVolume )
        return unexpected( std::move( maybeSimpleVolume.error() ) );

    TeethMaskToDirectionVolumeConvertor res;
    res.mask_ = std::move( *maybeSimpleVolume );

    HashSet<int> ids;
    for ( int i = 0; i < 49; ++i )
        if ( DentalId::fromFDI( i ) )
            ids.insert( i );
    for ( int i : additionalIds )
        ids.insert( i );

    HashSet<int> present;
    HashMap<int, Box3i> bounds;
    forEachInSimpleVolume( res.mask_, [&ids, &present, &bounds] ( const Vector3i& pos, float val )
    {
        const int vali = static_cast<int>( val );
        if ( ids.contains( vali ) )
        {
            present.insert( vali );
            bounds[vali].include( pos );
        }
    } );

    for ( int i : ids )
    {
        if ( present.contains( i ) )
        {
            res.presentObjects_[i] = bounds[i];
        }
    }

    return res;
}

Expected<TeethMaskToDirectionVolumeConvertor::ProcessResult> TeethMaskToDirectionVolumeConvertor::convertObject( int id ) const
{
    Box3i box;

    if ( auto it = presentObjects_.find( id ); it == presentObjects_.end() )
        return unexpected( fmt::format( "The mask does not contain specified object: {}", id ) );
    else
        box = it->second;


    const VolumeIndexer maskIndexer( mask_.dims );

    SimpleVolumeMinMax toothMask;
    toothMask.dims = box.size();
    toothMask.voxelSize = mask_.voxelSize;
    toothMask.data.resize( box.volume() );

    forEachInSimpleVolume( toothMask, [&box, &fullMask = mask_, &maskIndexer, &id] ( const auto& pt, float& v )
    {
        const auto fullPt = pt + box.min;
        if ( fullMask.data[maskIndexer.toVoxelId( fullPt )] == id )
            v = ( float )id;
        else
            v = 0;
    } );

    std::tie( toothMask.min, toothMask.max ) = parallelMinMax( toothMask.data );

    auto toothMaskVdb = simpleVolumeToVdbVolume( toothMask );

    return
        gridToMesh( toothMaskVdb.data, GridToMeshSettings{
            .voxelSize = toothMask.voxelSize,
            .isoValue = static_cast<float>( id ) - 0.001f
        } )
            .and_then( [voxelSize = toothMask.voxelSize, boxV = box] ( Mesh&& mesh )
            {
                Box3f box( mult( Vector3f( boxV.min ), voxelSize ), mult( Vector3f( boxV.max ), voxelSize ) );
                const auto xf = AffineXf3f::translation( box.min + voxelSize );

                MeshToDirectionVolumeParams p;
                p.vol.origin = box.min;
                p.vol.voxelSize = voxelSize;
                p.vol.dimensions = boxV.size();
                p.dist.maxDistSq = 1;
                p.dist.minDistSq = 0;
                p.projector = std::make_shared<PointsToMeshProjector>();

                mesh.transform( xf );

                p.projector->updateMeshData( &mesh );
                return meshToDirectionVolume( p )
                    .transform( [&xf] ( DirectionVolume&& volumes )
                    {
                        return ProcessResult{
                            .volume = std::move( volumes ),
                            .xf = xf
                        };
                    } );
            } )
            .transform( [&] ( ProcessResult&& r )
            {
                for ( int i = 0; i < 3; ++i )
                {
                    for ( auto v = 0_vox; v < toothMask.data.endId(); ++v )
                    {
                        if ( toothMask.data[v] == 0 )
                            r.volume[i].data[v] = invalidValue;
                    }
                }
                return std::move( r );
            } );
}

Expected<TeethMaskToDirectionVolumeConvertor::ProcessResult> TeethMaskToDirectionVolumeConvertor::convertAll() const
{
    std::vector<ProcessResult> teeth;
    std::vector<Box3i> toothBoxes;
    for ( const auto& [k, v] : presentObjects_ )
    {
        if ( auto maybeRes = convertObject( k ) )
        {
            teeth.push_back( std::move( *maybeRes ) );
            toothBoxes.push_back( v );
        }
        else
            return unexpected( std::move( maybeRes.error() ) );
    }

    std::array<SimpleVolumeMinMax, 3> res;
    for ( int i = 0; i < 3; ++i )
    {
        auto& r = res[i];
        r.voxelSize = mask_.voxelSize;
        r.dims = mask_.dims;
        r.data.clear();
        r.data.resize( mask_.data.size(), invalidValue );

        const VolumeIndexer maskInd( r.dims );
        for ( size_t j = 0; j < teeth.size(); ++j )
        {
            const auto& box = toothBoxes[j];
            const auto& t = teeth[j].volume[i];
            forEachInSimpleVolume( t, [&box, &r, &maskInd] ( const auto& pt, float v )
            {
                const auto fullPt = pt + box.min;
                if ( v != invalidValue )
                    r.data[maskInd.toVoxelId( fullPt )] = v;
            } );
        }
    }

    return ProcessResult{
        .volume = res,
        .xf = AffineXf3f{}
    };
}

Expected<std::array<SimpleVolumeMinMax, 3>> teethMaskToDirectionVolume( const VdbVolume& volume, const std::vector<int>& additionalIds )
{
    return TeethMaskToDirectionVolumeConvertor::create( volume, additionalIds )
            .and_then( &TeethMaskToDirectionVolumeConvertor::convertAll )
            .transform( [] ( auto&& x ) { return x.volume; } );
}

}
