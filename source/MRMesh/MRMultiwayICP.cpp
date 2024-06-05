#include "MRMultiwayICP.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPointToPointAligningTransform.h"
#include "MRPointToPlaneAligningTransform.h"
#include "MRMultiwayAligningTransform.h"
#include "MRBitSetParallelFor.h"
#include "MRAABBTreeObjects.h"
#include "MRAABBTreeObjects.h"
#include <algorithm>

namespace MR
{

MultiwayICP::MultiwayICP( const Vector<MeshOrPointsXf, ObjId>& objects, float samplingVoxelSize ) :
    objs_{ objects }
{
    resamplePoints( samplingVoxelSize );
}

Vector<AffineXf3f, ObjId> MultiwayICP::calculateTransformations( ProgressCallback cb )
{
    float minDist = std::numeric_limits<float>::max();
    int badIterCount = 0;
    resultType_ = ICPExitType::MaxIterations;

    for ( iter_ = 1; iter_ <= prop_.iterLimit; ++iter_ )
    {
        const bool pt2pt = ( prop_.method == ICPMethod::Combined && iter_ < 3 )
            || prop_.method == ICPMethod::PointToPoint;
        
        bool res = doIteration_( !pt2pt );

        if ( iter_ == 1 && perIterationCb_ )
            perIterationCb_( 0 );

        if ( perIterationCb_ )
            perIterationCb_( iter_ );

        if ( !res )
        {
            resultType_ = ICPExitType::NotFoundSolution;
            break;
        }

        const float curDist = pt2pt ? getMeanSqDistToPoint() : getMeanSqDistToPlane();
        if ( prop_.exitVal > curDist )
        {
            resultType_ = ICPExitType::StopMsdReached;
            break;
        }

        // exit if several(3) iterations didn't decrease minimization parameter
        if ( curDist < minDist )
        {
            minDist = curDist;
            badIterCount = 0;
        }
        else
        {
            if ( badIterCount >= prop_.badIterStopCount )
            {
                resultType_ = ICPExitType::MaxBadIterations;
                break;
            }
            badIterCount++;
        }
        if ( !reportProgress( cb, float( iter_ ) / float( prop_.iterLimit ) ) )
            return {};
    }

    Vector<AffineXf3f, ObjId> res;
    res.resize( objs_.size() );
    for ( int i = 0; i < objs_.size(); ++i )
        res[ObjId( i )] = objs_[ObjId( i )].xf;
    return res;
}

void MultiwayICP::resamplePoints( float samplingVoxelSize )
{
    MR_TIMER;
    samplingSize_ = samplingVoxelSize;

    Vector<VertBitSet, ObjId> samplesPerObj( objs_.size() );

    ParallelFor( objs_, [&] ( ObjId ind )
    {
        const auto& obj = objs_[ind];
        samplesPerObj[ind] = *obj.obj.pointsGridSampling( samplingVoxelSize );
    } );

    reservePairs_( samplesPerObj );

    // only do something if cascade mode is required
    resampleLayers_();
}

float MultiwayICP::getMeanSqDistToPoint() const
{
    return tbb::parallel_deterministic_reduce( tbb::blocked_range( size_t( 0 ), objs_.size() * objs_.size() ), NumSum(),
        [&] ( const auto& range, NumSum curr )
    {
        for ( size_t r = range.begin(); r < range.end(); ++r )
        {
            size_t i = r % objs_.size();
            size_t j = r / objs_.size();
            if ( i == j )
                continue;
            curr = curr + MR::getSumSqDistToPoint( pairsPerObj_[ObjId( i )][ObjId( j )] );
        }
        return curr;
    }, [] ( auto a, auto b ) { return a + b; } ).rootMeanSqF();
}

// unified function for mean sq dist to point calculation
float calcMeanSqDistToPointEx( const auto& pairs, auto id )
{
    return tbb::parallel_deterministic_reduce( tbb::blocked_range( decltype( id )( 0 ), decltype( id )( pairs.size() ) ), NumSum(),
        [&] ( const auto& range, NumSum curr )
    {
        for ( auto i = range.begin(); i < range.end(); ++i )
        {
            if ( i == id )
                continue;
            curr = curr + MR::getSumSqDistToPoint( pairs[i][id] ) + MR::getSumSqDistToPoint( pairs[id][i] );
        }
        return curr;
    }, [] ( auto a, auto b ) { return a + b; } ).rootMeanSqF();
}

float MultiwayICP::getMeanSqDistToPoint( ObjId id ) const
{
    return calcMeanSqDistToPointEx( pairsPerObj_, id );
}

float MultiwayICP::getLayerMeanSqDistToPoint_( Layer l, GroupId id ) const
{
    return calcMeanSqDistToPointEx( pairsPerLayer_[l], id );
}

float MultiwayICP::getMeanSqDistToPlane() const
{
    return tbb::parallel_deterministic_reduce( tbb::blocked_range( size_t( 0 ), objs_.size() * objs_.size() ), NumSum(),
        [&] ( const auto& range, NumSum curr )
    {
        for ( size_t r = range.begin(); r < range.end(); ++r )
        {
            size_t i = r % objs_.size();
            size_t j = r / objs_.size();
            if ( i == j )
                continue;
            curr = curr + MR::getSumSqDistToPlane( pairsPerObj_[ObjId( i )][ObjId( j )] );
        }
        return curr;
    }, [] ( auto a, auto b ) { return a + b; } ).rootMeanSqF();
}

float MultiwayICP::getMeanSqDistToPlane( ObjId id ) const
{
    return tbb::parallel_deterministic_reduce( tbb::blocked_range( ObjId( 0 ), ObjId( objs_.size() ) ), NumSum(),
        [&] ( const auto& range, NumSum curr )
    {
        for ( ObjId i = range.begin(); i < range.end(); ++i )
        {
            if ( i == id )
                continue;
            curr = curr + MR::getSumSqDistToPlane( pairsPerObj_[i][id] ) + MR::getSumSqDistToPlane( pairsPerObj_[id][i] );
        }
        return curr;
    }, [] ( auto a, auto b ) { return a + b; } ).rootMeanSqF();
}

size_t MultiwayICP::getNumActivePairs() const
{
    return tbb::parallel_deterministic_reduce( tbb::blocked_range( size_t( 0 ), objs_.size() * objs_.size() ), size_t( 0 ),
        [&] ( const auto& range, size_t curr )
    {
        for ( size_t r = range.begin(); r < range.end(); ++r )
        {
            size_t i = r % objs_.size();
            size_t j = r / objs_.size();
            if ( i == j )
                continue;
            curr += MR::getNumActivePairs( pairsPerObj_[ObjId( i )][ObjId( j )] );
        }
        return curr;
    }, [] ( auto a, auto b ) { return a + b; } );
}

size_t MultiwayICP::getNumActivePairs( ObjId id ) const
{
    return tbb::parallel_deterministic_reduce( tbb::blocked_range( ObjId( 0 ), ObjId( objs_.size() ) ), size_t( 0 ),
        [&] ( const auto& range, size_t curr )
    {
        for ( ObjId i = range.begin(); i < range.end(); ++i )
        {
            if ( i == id )
                continue;
            curr = curr + MR::getNumActivePairs( pairsPerObj_[i][id] ) + MR::getNumActivePairs( pairsPerObj_[id][i] );
        }
        return curr;
    }, [] ( auto a, auto b ) { return a + b; } );
}

std::string MultiwayICP::getStatusInfo() const
{
    return getICPStatusInfo( iter_, resultType_ );
}

void MultiwayICP::updateAllPointPairs()
{
    MR_TIMER;
    ParallelFor( size_t( 0 ), objs_.size() * objs_.size(), [&] ( size_t r )
    {
        auto i = ObjId( r % objs_.size() );
        auto j = ObjId( r / objs_.size() );
        if ( i == j )
            return;
        MR::updatePointPairs( pairsPerObj_[i][j], objs_[i], objs_[j], prop_.cosTreshold, prop_.distThresholdSq, prop_.mutualClosest );
    } );
    deactivateFarDistPairs_();
}

void MultiwayICP::reservePairs_( const Vector<VertBitSet, ObjId>& samplesPerObj )
{
    pairsPerObj_.clear();
    pairsPerObj_.resize( objs_.size() );
    for ( ObjId i( 0 ); i < objs_.size(); ++i )
    {
        auto& pairs = pairsPerObj_[i];
        pairs.resize( objs_.size() );
        bool groupWise = maxGroupSize_ > 1 && objs_.size() > maxGroupSize_;
        for ( ObjId j( 0 ); j < objs_.size(); ++j )
        {
            if ( i == j )
                continue;

            if ( groupWise )
            {
                auto groupI = i / maxGroupSize_;
                auto groupJ = j / maxGroupSize_;
                if ( groupI != groupJ )
                    continue;
            }

            auto& thisPairs = pairs[j];
            thisPairs.vec.reserve( samplesPerObj[i].count() );
            for ( auto v : samplesPerObj[i] )
                thisPairs.vec.emplace_back().srcVertId = v;
            thisPairs.active.reserve( thisPairs.vec.size() );
            thisPairs.active.clear();
        }
    }
}

void MultiwayICP::updatePointsPairsGroupWise_()
{
    MR_TIMER;
    ParallelFor( size_t( 0 ), objs_.size() * objs_.size(), [&] ( size_t r )
    {
        auto i = ObjId( r % objs_.size() );
        auto j = ObjId( r / objs_.size() );
        if ( i == j )
            return;
        auto groupI = i / maxGroupSize_;
        auto groupJ = j / maxGroupSize_;
        if ( groupI != groupJ )
            return;
        MR::updatePointPairs( pairsPerObj_[i][j], objs_[i], objs_[j], prop_.cosTreshold, prop_.distThresholdSq, prop_.mutualClosest );
    } );
    deactivateFarDistPairs_();
}

void MultiwayICP::resampleLayers_()
{
    MR_TIMER;
    if ( maxGroupSize_ <= 1 || maxGroupSize_ >= objs_.size() )
        return;

    int numLayers = 0;
    int numGroups = ( int( objs_.size() ) + maxGroupSize_ - 1 ) / maxGroupSize_;
    while ( numGroups > 1 )
    {
        numLayers++;
        numGroups = ( numGroups + maxGroupSize_ - 1 ) / maxGroupSize_;
    }

    Vector<Vector<MultiObjsSamples, GroupId>, Layer> samples( numLayers );
    int groupSize = 1;
    for ( Layer l = 0; l < numLayers; ++l )
    {
        groupSize *= maxGroupSize_;
        int layerSize = ( int( objs_.size() ) + groupSize - 1 ) / groupSize;
        auto& layerSamples = samples[l];
        layerSamples.resize( layerSize );
        ParallelFor( GroupId( 0 ), GroupId( layerSize ), [&] ( GroupId gId )
        {
            ObjId first = ObjId( groupSize * gId );
            ObjId last = ObjId( std::min( int( objs_.size() ), groupSize * ( gId + 1 ) ) );
            Vector<ModelPointsData, ObjId> groupData( last - first );
            for ( ObjId i( 0 ); i < groupData.size(); ++i )
            {
                const auto& obj = objs_[i + first];
                groupData[i] = { .points = &obj.obj.points(), .validPoints = &obj.obj.validPoints(),.xf = &obj.xf };
            }
            layerSamples[gId] = *multiModelGridSampling( groupData, samplingSize_ );
            for ( auto& smp : layerSamples[gId] )
                smp.objId += first;
        } );
    }
    reserveLayerPairs_( samples );
}

void MultiwayICP::reserveLayerPairs_( const Vector<Vector<MultiObjsSamples, GroupId>, Layer>& samples )
{
    MR_TIMER;
    pairsPerLayer_.resize( samples.size() );
    for ( Layer l( 0 ); l < pairsPerLayer_.size(); ++l )
    {
        const auto& samplesOnLayer = samples[l];
        auto& pairsOnLayer = pairsPerLayer_[l];
        int numGroups = int( samplesOnLayer.size() );
        pairsOnLayer.resize( numGroups );
        for ( GroupId gI( 0 ); gI < numGroups; ++gI )
        {
            auto& pairsOnGroup = pairsOnLayer[gI];
            pairsOnGroup.resize( numGroups );
            const auto& thisSamples = samplesOnLayer[gI];
            for ( GroupId gJ( 0 ); gJ < numGroups; ++gJ )
            {
                if ( gI == gJ )
                    continue;

                auto hyperGroupI = gI / maxGroupSize_;
                auto hyperGroupJ = gJ / maxGroupSize_;
                if ( hyperGroupI != hyperGroupJ )
                    continue;

                auto& thisPairs = pairsOnGroup[gJ];

                thisPairs.vec.resize( thisSamples.size() );
                for ( int i = 0; i < thisPairs.vec.size(); ++i )
                    thisPairs.vec[i].srcId = thisSamples[i];
                thisPairs.active.reserve( thisPairs.vec.size() );
                thisPairs.active.clear();
            }
        }
    }
}

void MultiwayICP::updateLayerPairs_( Layer l )
{
    MR_TIMER;
    int groupSize = 1;
    for ( int i = 0; i <= l; ++i )
        groupSize *= maxGroupSize_;
    auto numGroups = pairsPerLayer_[l].size();

    // build trees
    using GroupModels = Vector<MeshOrPointsXf, ObjId>;
    using LayerTrees = Vector<AABBTreeObjects, GroupId>;
    LayerTrees trees( numGroups );
    ParallelFor( trees, [&] ( GroupId gI )
    {
        auto gFirst = ObjId( gI * groupSize );
        auto gLast = ObjId( std::min( ( gI + 1 ) * groupSize, int( objs_.size() ) ) );
        
        GroupModels gObjs( begin( objs_ ) + gFirst, begin( objs_ ) + gLast );
        trees[gI] = AABBTreeObjects( std::move( gObjs ) );
    } );

    // update pairs
    auto gridSize = numGroups * numGroups;
    ParallelFor( 0, int( gridSize ), [&] ( int gridId )
    {
        GroupId gI = GroupId( gridId / numGroups );
        GroupId gJ = GroupId( gridId % numGroups );
        if ( gI == gJ )
            return;

        auto hyperGroupI = gI / maxGroupSize_;
        auto hyperGroupJ = gJ / maxGroupSize_;
        if ( hyperGroupI != hyperGroupJ )
            return;

        auto srcFirst = ObjId( gI * groupSize );
        auto& pairs = pairsPerLayer_[l][gI][gJ];
        pairs.active.resize( pairs.vec.size() );
        auto tgtFirst = ObjId( gJ * groupSize );
        BitSetParallelForAll( pairs.active, [&] ( size_t i )
        {
            pairs.active.set( i, projectGroupPair_( pairs.vec[i], trees[gI], srcFirst, trees[gJ], tgtFirst ) );
        } );
    } );
    deactivateFarDistLayerPairs_( l );
}

bool MultiwayICP::projectGroupPair_( GroupPair& pair, const AABBTreeObjects& srcTree, ObjId srcFirst, const AABBTreeObjects& tgtTree, ObjId tgtFirst )
{
    const auto& srcObjXf = objs_[pair.srcId.objId];
    const auto& srcPoint = srcObjXf.obj.points()[pair.srcId.vId];
    pair.srcPoint = srcObjXf.xf( srcPoint );

    ObjId minObjId{};
    MeshOrPoints::ProjectionResult minPrj{};
    minPrj.distSq = prop_.distThresholdSq;
    projectOnAll( pair.srcPoint, tgtTree, minPrj.distSq,
        [&] ( ObjId objId, MeshOrPoints::ProjectionResult prj )
    {
        if ( prj.distSq < minPrj.distSq )
        {
            minPrj = prj;
            minObjId = tgtFirst + objId;
        }
    } );

    if ( !minPrj.closestVert )
        return false;

    if ( minPrj.isBd )
        return false;

    const auto& tgtObjXf = objs_[minObjId];
    const auto srcNormals = srcObjXf.obj.normals();
    const auto tgtNormals = tgtObjXf.obj.normals();

    pair.srcNorm = srcNormals ? ( srcObjXf.xf.A * srcNormals( pair.srcId.vId ) ).normalized() : Vector3f();
    
    pair.tgtClosestId.objId = minObjId;
    pair.tgtClosestId.vId = minPrj.closestVert;
    pair.tgtPoint = tgtObjXf.xf( minPrj.point );
    pair.tgtNorm = minPrj.normal ? ( tgtObjXf.xf.A * minPrj.normal.value() ).normalized() : Vector3f();

    auto normalsAngleCos = ( minPrj.normal && srcNormals ) ? dot( pair.tgtNorm, pair.srcNorm ) : 1.0f;

    if ( normalsAngleCos < prop_.cosTreshold )
        return false;

    if ( prop_.mutualClosest )
    {
        ObjId mutalMinObjId{};
        MeshOrPoints::ProjectionResult mutalMinPrj{};
        mutalMinPrj.distSq = prop_.distThresholdSq;
        projectOnAll( pair.tgtPoint, srcTree, mutalMinPrj.distSq,
            [&] ( ObjId objId, MeshOrPoints::ProjectionResult prj )
        {
            if ( prj.distSq < mutalMinPrj.distSq )
            {
                mutalMinPrj = prj;
                mutalMinObjId = srcFirst + objId;
            }
        } );
        if ( !mutalMinPrj.closestVert )
            return false;
        if ( mutalMinObjId != pair.srcId.objId )
            return false;
        if ( mutalMinPrj.closestVert != pair.srcId.vId )
            return false;
    }

    const auto srcWeights = srcObjXf.obj.weights();
    pair.weight = srcWeights ? srcWeights( pair.srcId.vId ) : 1.0f;
    pair.distSq = minPrj.distSq;

    return true;
}

// unified function do disable far pairs
template<typename T, typename IndexT>
void deactivateFarDistPairsEx( Vector<T, IndexT>& pairs, float distThresholdSq, float farDistFactor )
{
    MR_TIMER;
    Vector<float, IndexT> maxDistSq( pairs.size() );
    for ( int it = 0; it < 3; ++it )
    {
        ParallelFor( maxDistSq, [&] ( auto id )
        {
            maxDistSq[id] = sqr( farDistFactor * MR::calcMeanSqDistToPointEx( pairs, id ) );
        } );

        tbb::enumerable_thread_specific<size_t> counters( 0 );
        ParallelFor( size_t( 0 ), pairs.size() * pairs.size(), [&] ( size_t r )
        {
            auto i = IndexT( r % pairs.size() );
            auto j = IndexT( r / pairs.size() );
            if ( i == j )
                return;
            if ( maxDistSq[i] >= distThresholdSq )
                return;
            counters.local() += (
                MR::deactivateFarPairs( pairs[i][j], maxDistSq[i] ) +
                MR::deactivateFarPairs( pairs[j][i], maxDistSq[i] ) );
        } );

        size_t numDeactivated = 0;
        for ( auto counter : counters )
            numDeactivated += counter;
        if ( numDeactivated == 0 )
            break;
    }
}

void MultiwayICP::deactivateFarDistPairs_()
{
    MR::deactivateFarDistPairsEx( pairsPerObj_, prop_.distThresholdSq, prop_.farDistFactor );
}

void MultiwayICP::deactivateFarDistLayerPairs_( Layer l )
{
    MR::deactivateFarDistPairsEx( pairsPerLayer_[l], prop_.distThresholdSq, prop_.farDistFactor );
}

bool MultiwayICP::doIteration_( bool p2pl )
{
    if ( maxGroupSize_ == 1 )
    {
        updateAllPointPairs();
        return p2pl ? p2plIter_() : p2ptIter_();
    }
    if ( maxGroupSize_ < 1 || maxGroupSize_ >= objs_.size() )
    {
        updateAllPointPairs();
        return multiwayIter_( p2pl );
    }
    // pair updated inside
    return multiwayIter_( maxGroupSize_, p2pl );
}

bool MultiwayICP::p2ptIter_()
{
    MR_TIMER;
    using FullSizeBool = uint8_t;
    Vector<FullSizeBool, ObjId> valid( objs_.size() );
    ParallelFor( objs_, [&] ( ObjId id )
    {
        PointToPointAligningTransform p2pt;
        for ( ObjId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairsPerObj_[id][j].active )
            {
                const auto& vp = pairsPerObj_[id][j].vec[idx];
                p2pt.add( vp.srcPoint, 0.5f * ( vp.srcPoint + vp.tgtPoint ), vp.weight );
            }
            for ( size_t idx : pairsPerObj_[j][id].active )
            {
                const auto& vp = pairsPerObj_[j][id].vec[idx];
                p2pt.add( vp.tgtPoint, 0.5f * ( vp.srcPoint + vp.tgtPoint ), vp.weight );
            }
        }

        AffineXf3f res;
        switch ( prop_.icpMode )
        {
        default:
            assert( false );
            [[fallthrough]];
        case ICPMode::RigidScale:
            res = AffineXf3f( p2pt.findBestRigidScaleXf() );
            break;
        case ICPMode::AnyRigidXf:
            res = AffineXf3f( p2pt.findBestRigidXf() );
            break;
        case ICPMode::OrthogonalAxis:
            res = AffineXf3f( p2pt.findBestRigidXfOrthogonalRotationAxis( Vector3d{ prop_.fixedRotationAxis } ) );
            break;
        case ICPMode::FixedAxis:
            res = AffineXf3f( p2pt.findBestRigidXfFixedRotationAxis( Vector3d{ prop_.fixedRotationAxis } ) );
            break;
        case ICPMode::TranslationOnly:
            res = AffineXf3f( Matrix3f(), Vector3f( p2pt.findBestTranslation() ) );
            break;
        }
        if ( std::isnan( res.b.x ) ) //nan check
        {
            valid[id] = FullSizeBool( false );
            return;
        }
        objs_[id].xf = res * objs_[id].xf;
        valid[id] = FullSizeBool( true );
    } );

    return std::all_of( valid.vec_.begin(), valid.vec_.end(), [] ( auto v ) { return bool( v ); } );
}

bool MultiwayICP::p2plIter_()
{
    MR_TIMER;
    using FullSizeBool = uint8_t;
    Vector<FullSizeBool, ObjId> valid( objs_.size() );
    ParallelFor( objs_, [&] ( ObjId id )
    {
        Vector3f centroidRef;
        int activeCount = 0;
        for ( ObjId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairsPerObj_[id][j].active )
            {
                const auto& vp = pairsPerObj_[id][j].vec[idx];
                centroidRef += ( vp.tgtPoint + vp.srcPoint ) * 0.5f;
                centroidRef += vp.srcPoint;
                ++activeCount;
            }
            for ( size_t idx : pairsPerObj_[j][id].active )
            {
                const auto& vp = pairsPerObj_[j][id].vec[idx];
                centroidRef += ( vp.tgtPoint + vp.srcPoint ) * 0.5f;
                centroidRef += vp.tgtPoint;
                ++activeCount;
            }
        }
        if ( activeCount <= 0 )
        {
            valid[id] = FullSizeBool( false );
            return;
        }
        centroidRef /= float( activeCount * 2 );
        AffineXf3f centroidRefXf = AffineXf3f( Matrix3f(), centroidRef );

        PointToPlaneAligningTransform p2pl;
        for ( ObjId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairsPerObj_[id][j].active )
            {
                const auto& vp = pairsPerObj_[id][j].vec[idx];
                p2pl.add( vp.srcPoint - centroidRef, ( vp.tgtPoint + vp.srcPoint ) * 0.5f - centroidRef, vp.tgtNorm, vp.weight );
            }
            for ( size_t idx : pairsPerObj_[j][id].active )
            {
                const auto& vp = pairsPerObj_[j][id].vec[idx];
                p2pl.add( vp.tgtPoint - centroidRef, ( vp.tgtPoint + vp.srcPoint ) * 0.5f - centroidRef, vp.srcNorm, vp.weight );
            }
        }
        p2pl.prepare();

        AffineXf3f res = getAligningXf( p2pl, prop_.icpMode, prop_.p2plAngleLimit, prop_.p2plScaleLimit, prop_.fixedRotationAxis );
        if ( std::isnan( res.b.x ) ) //nan check
        {
            valid[id] = FullSizeBool( false );
            return;
        }
        objs_[id].xf = centroidRefXf * res * centroidRefXf.inverse() * objs_[id].xf;
        valid[id] = FullSizeBool( true );
    } );
    return std::all_of( valid.vec_.begin(), valid.vec_.end(), [] ( auto v ) { return bool( v ); } );
}

bool MultiwayICP::multiwayIter_( bool p2pl )
{
    MR_TIMER;
    Vector<MultiwayAligningTransform,ObjId> mats( objs_.size() );
    ParallelFor( mats, [&] ( ObjId i )
    {
        auto& mat = mats[i];
        mat.reset( int( objs_.size() ) );
        for ( ObjId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == i )
                continue;
            for ( auto idx : pairsPerObj_[i][j].active )
            {
                const auto& data = pairsPerObj_[i][j].vec[idx];
                if ( p2pl )
                    mat.add( int( i ), data.srcPoint, int( j ), data.tgtPoint, ( data.tgtNorm + data.srcNorm ).normalized(), data.weight );
                else
                    mat.add( int( i ), data.srcPoint, int( j ), data.tgtPoint, data.weight );
            }
        }
    } );

    MultiwayAligningTransform mat;
    mat.reset( int( objs_.size() ) );
    for ( const auto& m : mats )
        mat.add( m );

    mats = {}; // free memory

    MultiwayAligningTransform::Stabilizer stabilizer;
    stabilizer.rot = samplingSize_ * 1e-1f;
    stabilizer.shift = 1e-3f;
    auto res = mat.solve( stabilizer );
    for ( ObjId i( 0 ); i < objs_.size(); ++i )
    {
        auto resI = res[i.get()].rigidXf();
        if ( std::isnan( resI.b.x ) )
            return false;
        objs_[i].xf = AffineXf3f( resI * AffineXf3d( objs_[i].xf ) );
    }
    return true;
}

bool MultiwayICP::multiwayIter_( int groupSize, bool p2pl /*= true */ )
{
    int numLayer = -1; // trivial layer
    int subGroupSize = 1;
    for ( ;;)
    {
        int grainSize = subGroupSize * groupSize;
        int numGrains = ( int( objs_.size() ) + grainSize - 1 ) / grainSize;

        if ( subGroupSize == 1 )
            updatePointsPairsGroupWise_();
        else
            updateLayerPairs_( numLayer );

        for ( int gId = 0; gId < numGrains; ++gId )
        {
            int groupFirst = gId * grainSize;
            int groupsLastEx = std::min( groupFirst + grainSize, int( objs_.size() ) );
            assert( groupsLastEx > groupFirst );
            if ( groupsLastEx == groupFirst + 1 )
                continue;

            int numSubGroups = ( ( groupsLastEx - groupFirst ) + subGroupSize - 1 ) / subGroupSize;
            if ( numSubGroups == 1 )
                continue;

            MultiwayAligningTransform mat;
            mat.reset( numSubGroups );
            for ( GroupId sbI( 0 ); sbI < numSubGroups; ++sbI )
            for ( GroupId sbJ( 0 ); sbJ < numSubGroups; ++sbJ )
            {
                if ( sbI == sbJ )
                    continue;
                if ( numLayer < 0 )
                {
                    ObjId iBegin = ObjId( groupFirst + sbI * subGroupSize );
                    ObjId iEnd = ObjId( std::min( groupFirst + ( sbI + 1 ) * subGroupSize, groupsLastEx ) );
                    ObjId jBegin = ObjId( groupFirst + sbJ * subGroupSize );
                    ObjId jEnd = ObjId( std::min( groupFirst + ( sbJ + 1 ) * subGroupSize, groupsLastEx ) );
                    for ( ObjId i( iBegin ); i < iEnd; ++i )
                    for ( ObjId j( jBegin ); j < jEnd; ++j )
                    {
                        for ( auto idx : pairsPerObj_[i][j].active )
                        {
                            const auto& data = pairsPerObj_[i][j].vec[idx];
                            if ( p2pl )
                                mat.add( sbI, data.srcPoint, sbJ, data.tgtPoint, ( data.tgtNorm + data.srcNorm ).normalized(), data.weight );
                            else
                                mat.add( sbI, data.srcPoint, sbJ, data.tgtPoint, data.weight );
                        }
                    }
                }
                else
                {
                    auto globalGroupI = GroupId( gId * groupSize + sbI );
                    auto globalGroupJ = GroupId( gId * groupSize + sbJ );
                    const auto& pairs = pairsPerLayer_[numLayer][globalGroupI][globalGroupJ];
                    for ( auto idx : pairs.active )
                    {
                        const auto& data = pairs.vec[idx];
                        if ( p2pl )
                            mat.add( sbI, data.srcPoint, sbJ, data.tgtPoint, ( data.tgtNorm + data.srcNorm ).normalized(), data.weight );
                        else
                            mat.add( sbI, data.srcPoint, sbJ, data.tgtPoint, data.weight );
                    }
                }
            }

            MultiwayAligningTransform::Stabilizer stabilizer;
            stabilizer.rot = samplingSize_ * 1e-1f;
            stabilizer.shift = 1e-3f;
            auto res = mat.solve( stabilizer );
            for ( int sbI = 0; sbI < numSubGroups; ++sbI )
            {
                auto resI = res[sbI].rigidXf();
                if ( std::isnan( resI.b.x ) )
                    return false;
                ObjId iBegin = ObjId( groupFirst + sbI * subGroupSize );
                ObjId iEnd = ObjId( std::min( groupFirst + ( sbI + 1 ) * subGroupSize, groupsLastEx ) );
                for ( ObjId i( iBegin ); i < iEnd; ++i )
                    objs_[i].xf = AffineXf3f( resI * AffineXf3d( objs_[i].xf ) );
            }
        }
        if ( numGrains == 1 )
            break;
        subGroupSize *= groupSize;
        ++numLayer;
    }
    return true;
}

}
