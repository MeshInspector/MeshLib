#include "MRTetrisNesting.h"
#include "MRMesh/MRBitSet.h"
#include "MRVoxels/MRDistanceVolumeParams.h"
#include "MRMesh/MRParallelFor.h"
#include "MRMesh/MRUnionFind.h"
#include "MRMesh/MRBitSetParallelFor.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MRBox.h"
#include "MRVoxels/MRCalcDims.h"
#include "MRVoxels/MRMeshToDistanceVolume.h"
#include "MRVoxels/MRVoxelsVolume.h"
#include "MRVoxels/MRVoxelsSave.h"
#include <limits>

namespace MR
{

namespace Nesting
{

class GroupTag;
using GId = Id<GroupTag>;

struct GroupsInfo
{
    Vector<GId, ObjId> o2gId;
    Vector<ObjBitSet, GId> groups;
};

// we use int16_t, so we don't expect large shifts
class MinShiftsMatrix
{
public:
    MinShiftsMatrix( size_t numObjs ) : numObjs_{ numObjs }
    {
        minShifts_.resize( numObjs_ * numObjs_, std::numeric_limits<int16_t>::max() );
    }

    size_t numObjects() const
    {
        return numObjs_;
    }

    int16_t get( GId a, GId b ) const
    {
        return minShifts_[a * numObjs_ + b];
    }
    int16_t& get( GId a, GId b )
    {
        return minShifts_[a * numObjs_ + b];
    }

    void setMin( GId a, GId b, int16_t val )
    {
        auto& ownval = get( a, b );
        if ( val < ownval )
            ownval = val;
    }

    void mergeOther( const MinShiftsMatrix& other )
    {
        assert( other.numObjs_ == numObjs_ );
        for ( int i = 0; i < minShifts_.size(); ++i )
            minShifts_[i] = std::min( minShifts_[i], other.minShifts_[i] );
    }

    Vector<int16_t, GId> process()
    {
        Vector<int16_t, GId> shifts( numObjs_, 0 );
        bool shiftsAdded = true;
        while ( shiftsAdded )
        {
            shiftsAdded = false;
            for ( GId a( 0 ); a < numObjs_; ++a )
            {
                auto ms = findMinShift_( a );
                if ( ms == std::numeric_limits<int16_t>::max() || ms == 0 )
                    continue;
                shift_( a, ms );
                shifts[a] += ms;
                shiftsAdded = true;
            }
        }
        return shifts;
    }
private:
    int16_t findMinShift_( GId a ) const
    {
        int16_t minVal = std::numeric_limits<int16_t>::max();
        for ( GId b( 0 ); b < numObjs_; ++b )
        {
            auto val = get( a, b );
            if ( val < minVal )
                minVal = val;
        }
        return minVal;
    }

    void shift_( GId a, int16_t val )
    {
        for ( GId b( 0 ); b < numObjs_; ++b )
        {
            auto& ownval = get( a, b );
            if ( ownval != std::numeric_limits<int16_t>::max() )
                ownval -= val;
        }
        for ( GId b( 0 ); b < numObjs_; ++b )
        {
            auto& ownval = get( b, a );
            if ( a != b && ownval != std::numeric_limits<int16_t>::max() )
                ownval += val;
        }
    }

    std::vector<int16_t> minShifts_;
    size_t numObjs_{ 0 };
};

static std::pair<GroupsInfo, MinShiftsMatrix> convergeShiftsMatrix( MinShiftsMatrix&& sm, const GroupsInfo* optPrevLayerGroup = nullptr )
{
    UnionFind<ObjId> uf;
    uf.reset( sm.numObjects() );
    for ( int a = 0; a < sm.numObjects(); ++a )
    {
        for ( int b = 0; b < sm.numObjects(); ++b )
        {
            if ( sm.get( GId( a ), GId( b ) ) == 0 )
            {
                uf.unite( ObjId( a ), ObjId( b ) );
            }
        }
    }

    GroupsInfo resG;
    resG.o2gId.resize( sm.numObjects() );
    for ( ObjId a( 0 ); a < sm.numObjects(); ++a )
    {
        auto& agId = resG.o2gId[a];
        if ( !agId )
        {
            agId = GId( resG.groups.size() );
            resG.groups.emplace_back();
        }
        resG.groups[agId].autoResizeSet( a );
        for ( ObjId b( a + 1 ); b < sm.numObjects(); ++b )
        {
            if ( !uf.united( a, b ) )
                continue;
            resG.o2gId[b] = agId;
            resG.groups[agId].autoResizeSet( b );
        }
    }

    MinShiftsMatrix resM( resG.groups.size() );
    for ( GId ag( 0 ); ag < resM.numObjects(); ++ag )
    {
        for ( GId bg( 0 ); bg < resM.numObjects(); ++bg )
        {
            int16_t minVal = std::numeric_limits<int16_t>::max();
            for ( auto a : resG.groups[ag] )
            {
                if ( ag != bg )
                {
                    for ( auto b : resG.groups[bg] )
                    {
                        minVal = std::min( minVal, sm.get( GId( int( a ) ), GId( int( b ) ) ) );
                    }
                }
                else
                    minVal = std::min( minVal, sm.get( GId( int( a ) ), GId( int( a ) ) ) );
            }
            resM.setMin( ag, bg, minVal );
        }
    }

    if ( optPrevLayerGroup )
    {
        // update groups to gave bitsets of original objects instead of bitsets of prev layer groups
        ObjBitSet accumBitSet;
        for ( auto& group : resG.groups )
        {
            accumBitSet.reset();
            for ( auto gId : group )
            {
                accumBitSet |= optPrevLayerGroup->groups[optPrevLayerGroup->o2gId[gId]];
            }
            group = accumBitSet;
        }
    }
    return std::make_pair( std::move( resG ), std::move( resM ) );
}

struct TetrisBlock
{
    VoxelBitSet objectVoxels;
    Vector3i dims;
    Vector3i position;
};

class TetrisDensifier
{
public:
    TetrisDensifier( const Vector<MeshXf, ObjId>& meshes, const TetrisDensifyParams& params );

    Expected<Vector<AffineXf3f, ObjId>> run();
private:
    TetrisDensifyParams params_;

    Vector<TetrisBlock, ObjId> tetrisBlocks_;

    Vector<ObjId, VoxelId> nestVoxels_;
    Vector3i nestDims_;

    void buildBlocks_( const Vector<MeshXf, ObjId>& meshes );
    bool fillNest_( const ProgressCallback& cb );
    MinShiftsMatrix calcInitShifts_( OutEdge dir ) const;
    Vector<int16_t, ObjId> calcResShifts_( MinShiftsMatrix&& sm ) const;
};

TetrisDensifier::TetrisDensifier( const Vector<MeshXf, ObjId>& meshes, const TetrisDensifyParams& params ) :
    params_{ params }
{
    auto dims = Vector3i( params_.baseParams.nest.size() / params_.options.voxelSize ) + Vector3i::diagonal( 1 );
    if ( params_.options.nestDimensionsCache )
    {
        if ( *params_.options.nestDimensionsCache != Vector3i() )
            dims = *params_.options.nestDimensionsCache; // if not zero - take input dimensions
        else
            *params_.options.nestDimensionsCache = dims; // if zero - output calculated to `caches`
    }
    if ( params_.options.nestVoxelsCache )
    {
        nestVoxels_ = std::move( *params_.options.nestVoxelsCache ); // move it into class for further usage
    }

    nestDims_ = dims;
    buildBlocks_( meshes );
}

Expected<Vector<AffineXf3f, ObjId>> TetrisDensifier::run()
{
    MR_TIMER;

    auto sb = subprogress( params_.options.cb, 0.5f, 1.0f );
    Vector<AffineXf3f, ObjId> combinedShifts( tetrisBlocks_.size() );
    int numSteps = int( params_.options.densificationSequence.size() );
    for ( int i = 0; i < numSteps; ++i )
    {
        OutEdge dir = params_.options.densificationSequence[i];

        if ( !fillNest_( subprogress( sb, float( i ) / float( numSteps ), float( i + 1 ) / float( numSteps ) ) ) )
            return unexpectedOperationCanceled();

        // debug output voxels
#if 0
        static int numSave = 0;
        {
            SimpleVolume sv;
            sv.data.resize( nestVoxels_.size() );
            ParallelFor( nestVoxels_, [&] ( VoxelId v )
            {
                sv.data[v] = float( int( nestVoxels_[v] ) );
            } );
            sv.dims = nestDims_;
            sv.voxelSize = Vector3f::diagonal( params_.options.voxelSize );
            ( void ) VoxelsSave::toRawAutoname( sv, "D:\\WORK\\MODELS\\Nesting\\DEBUG\\" + std::to_string( ++numSave ) + "_vox.raw" );
        }
#endif


        auto shifts = calcResShifts_( calcInitShifts_( dir ) );
        for ( ObjId o( 0 ); o < combinedShifts.size(); ++o )
        {
            auto addShift = neiPosDelta[int( dir )] * int( shifts[o] );
            tetrisBlocks_[o].position += addShift;
            combinedShifts[o].b += Vector3f( addShift ) * params_.options.voxelSize;
        }
    }

    // output occupied voxels
    if ( params_.options.occupiedVoxelsCache )
    {
        params_.options.occupiedVoxelsCache->resize( nestVoxels_.size() );
        BitSetParallelForAll( *params_.options.occupiedVoxelsCache, [&] ( VoxelId vid )
        {
            params_.options.occupiedVoxelsCache->set( vid, bool( nestVoxels_[vid] ) );
        } );
    }
    // output nest voxels
    if ( params_.options.nestVoxelsCache )
        *params_.options.nestVoxelsCache = std::move( nestVoxels_ ); // return it for possible subsequent calls

    return combinedShifts;
}

void TetrisDensifier::buildBlocks_( const Vector<MeshXf, ObjId>& meshes )
{
    MR_TIMER;
    tetrisBlocks_.resize( meshes.size() );
    ParallelFor( meshes, [&] ( ObjId oId )
    {
        if ( !meshes[oId].mesh )
            return;
        auto box = meshes[oId].mesh->computeBoundingBox( &meshes[oId].xf );
        box = box.expanded( Vector3f::diagonal( params_.baseParams.minInterval ) );
        auto [org, dims] = calcOriginAndDimensions( box, params_.options.voxelSize );
        tetrisBlocks_[oId].position = Vector3i( ( org - params_.baseParams.nest.min ) / params_.options.voxelSize );
        org = params_.baseParams.nest.min + Vector3f( tetrisBlocks_[oId].position ) * params_.options.voxelSize;

        CloseToMeshVolumeParams tParams;
        // half min interval, so two objects together have 0.5+0.5=1
        tParams.closeDist = std::max( 0.5f * params_.baseParams.minInterval, params_.options.voxelSize * std::sqrt( 3.0f ) * 0.5f ); // half voxel diagonal for small intervals
        tParams.meshToWorld = &meshes[oId].xf;
        tParams.vol.dimensions = dims;
        tParams.vol.origin = org;
        tParams.vol.voxelSize = Vector3f::diagonal( params_.options.voxelSize );

        auto res = makeCloseToMeshVolume( *meshes[oId].mesh, tParams );
        if ( !res.has_value() )
        {
            assert( false );
            return;
        }
        tetrisBlocks_[oId].dims = dims;
        tetrisBlocks_[oId].objectVoxels = std::move( res->data );
    }, subprogress( params_.options.cb, 0.0f, 0.5f ) );
}

bool TetrisDensifier::fillNest_( const ProgressCallback& cb )
{
    MR_TIMER;
    if ( !reportProgress( cb, 0.0f ) )
        return false;

    if ( nestVoxels_.empty() )
    {
        nestVoxels_.resize( size_t( nestDims_.z ) * nestDims_.y * nestDims_.x, ObjId() );
    }
    else
    {
        std::fill( nestVoxels_.vec_.begin(), nestVoxels_.vec_.end(), ObjId() );
    }

    if ( !reportProgress( cb, 0.1f ) )
        return false;

    auto sb2 = subprogress( cb, 0.1f, 0.9f );

    auto nestIndexer = VolumeIndexer( nestDims_ );
    for ( ObjId o( 0 ); o < tetrisBlocks_.size(); ++o )
    {
        auto localIndexer = VolumeIndexer( tetrisBlocks_[o].dims );
        BitSetParallelFor( tetrisBlocks_[o].objectVoxels, [&] ( VoxelId v )
        {
            auto nestPos = localIndexer.toPos( v ) + tetrisBlocks_[o].position;
            if ( nestPos.x < 0 || nestPos.x >= nestDims_.x ||
                 nestPos.y < 0 || nestPos.y >= nestDims_.y ||
                 nestPos.z < 0 || nestPos.z >= nestDims_.z )
                return;
            nestVoxels_[nestIndexer.toVoxelId( nestPos )] = o;
        } );
        if ( !reportProgress( sb2, float( int( o ) + 1 ) / float( tetrisBlocks_.size() ) ) )
            return false;
    }

    // fill external second to improve "slide-ability"
    auto invalidObjId = ObjId( tetrisBlocks_.size() );
    if ( params_.options.occupiedVoxelsCache && params_.options.occupiedVoxelsCache->any() )
    {
        BitSetParallelFor( *params_.options.occupiedVoxelsCache, [&] ( VoxelId vid )
        {
            nestVoxels_[vid] = invalidObjId;
        } );
    }
#if 0 // at one point we decided that there is no need in safe layers
    else
    {
        int numSafeLayers = int( params_.baseParams.minInterval * 0.5f / params_.options.voxelSize );
        Vector3i safeDims = nestDims_ - Vector3i::diagonal( numSafeLayers );
        VolumeIndexer indexer( nestDims_ );
        ParallelFor( nestVoxels_, [&] ( VoxelId vid )
        {
            auto pos = indexer.toPos( vid );
            if ( pos.x < numSafeLayers || pos.y < numSafeLayers || pos.z < numSafeLayers ||
                 pos.x >= safeDims.x || pos.y >= safeDims.y || pos.z >= safeDims.z )
                nestVoxels_[vid] = invalidObjId;
        } );
    }
#endif
    if ( !reportProgress( cb, 1.0f ) )
        return false;

    return true;
}

MinShiftsMatrix TetrisDensifier::calcInitShifts_( OutEdge dir ) const
{
    MR_TIMER;
    VolumeIndexer indexer( nestDims_ );
    size_t numobjs = tetrisBlocks_.size();
    auto invalidObjId = ObjId( numobjs );
    tbb::enumerable_thread_specific<MinShiftsMatrix> shiftsTLS( numobjs );
    ParallelFor( nestVoxels_, shiftsTLS, [&] ( VoxelId voxel, MinShiftsMatrix& tls )
    {
        auto thisObjId = nestVoxels_[voxel];
        if ( !thisObjId || thisObjId == invalidObjId )
            return;
        VoxelId nextVox = voxel;
        ObjId nextObjId;
        int16_t shift = 0;
        for ( ;; )
        {
            nextVox = indexer.getNeighbor( nextVox, dir );
            if ( !nextVox )
                break;
            nextObjId = nestVoxels_[nextVox];
            if ( !nextObjId )
                ++shift; // free space
            else if ( nextObjId == thisObjId )
                return; // same object
            else // other object
                break;
        }
        if ( !nextObjId || nextObjId == invalidObjId )
            nextObjId = thisObjId; // this means nest wall or cache->occupiedVoxel
        tls.setMin( GId( int( thisObjId ) ), GId( int( nextObjId ) ), shift );
    } );
    MinShiftsMatrix res( numobjs );
    for ( const auto& shiftTLS : shiftsTLS )
        res.mergeOther( shiftTLS );
    return res;
}

Vector<int16_t, ObjId> TetrisDensifier::calcResShifts_( MinShiftsMatrix&& sm ) const
{
    MR_TIMER;
    size_t numobjs = tetrisBlocks_.size();

    Vector<int16_t, ObjId> res( numobjs, 0 );

    auto gsm = convergeShiftsMatrix( std::move( sm ), nullptr );

    for ( ;; )
    {
        bool atLeastOneChanged = false;
        auto gShifts = gsm.second.process();
        for ( GId i( 0 ); i < gShifts.size(); ++i )
        {
            auto shift = gShifts[i];
            if ( shift == 0 )
                continue;
            atLeastOneChanged = true;
            for ( auto objId : gsm.first.groups[i] )
                res[objId] += shift;
        }
        if ( !atLeastOneChanged )
            break;
        gsm = convergeShiftsMatrix( std::move( gsm.second ), &gsm.first );
    }
    return res;
}

Expected<Vector<AffineXf3f, ObjId>> tetrisNestingDensify( const Vector<MeshXf, ObjId>& meshes, const TetrisDensifyParams& params )
{
    MR_TIMER;
    if ( params.options.densificationSequence.empty() )
    {
        assert( false );
        return unexpected( "Empty densification sequence" );
    }
    Nesting::TetrisDensifier td( meshes, params );
    return td.run();
}

}

}
