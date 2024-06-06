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

void updateGroupPairs( ICPGroupPairs& pairs, const ICPObjects& objs,
    ICPGroupProjector srcProjector, ICPGroupProjector tgtProjector, 
    float cosTreshold, float distThresholdSq, bool mutualClosest )
{
    MR_TIMER;
    pairs.active.clear();
    pairs.active.resize( pairs.vec.size(), true );

    // calculate pairs
    BitSetParallelForAll( pairs.active, [&] ( size_t idx )
    {
        auto & res = pairs.vec[idx];
        const auto p0 = objs[res.srcId.objId].obj.points()[res.srcId.vId];
        const auto pt = objs[res.srcId.objId].xf( p0 );

        ObjId prjObj;
        MeshOrPoints::ProjectionResult prj;
        // do not search for target point further than distance threshold
        prj.distSq = distThresholdSq;
        if ( res.tgtClosestId.objId && res.tgtClosestId.vId )
        {
            // start with old closest point ...
            prj.point = objs[res.tgtClosestId.objId].obj.points()[res.tgtClosestId.vId];
            if ( auto tgtNormals = objs[res.tgtClosestId.objId].obj.normals() )
                prj.normal = tgtNormals( res.tgtClosestId.vId );
            prj.distSq = ( pt - objs[res.tgtClosestId.objId].xf( prj.point ) ).lengthSq();
            prj.closestVert = res.tgtClosestId.vId;
            prjObj = res.tgtClosestId.objId;
        }
        // ... and try to find only closer one
        tgtProjector( pt, prj, prjObj );
        if ( !prj.closestVert || prj.distSq > distThresholdSq || prj.isBd )
        {
            // no target point found within distance threshold
            pairs.active.reset( idx );
            return;
        }
        assert( prjObj );
        const auto p1 = prj.point;

        res.distSq = prj.distSq;
        res.weight = 1.0;
        if ( auto srcWeights = objs[res.srcId.objId].obj.weights() )
            res.weight = srcWeights( res.srcId.vId );

        res.tgtClosestId.objId = prjObj;
        res.tgtClosestId.vId = prj.closestVert;

        res.srcPoint = pt;
        res.tgtPoint = objs[res.tgtClosestId.objId].xf( prj.point );

        auto srcNormals = objs[res.srcId.objId].obj.normals();
        res.srcNorm = srcNormals ? ( objs[res.srcId.objId].xf.A * srcNormals( res.srcId.vId ) ).normalized() : Vector3f();
        res.tgtNorm = prj.normal ? ( objs[res.tgtClosestId.objId].xf.A * prj.normal.value() ).normalized() : Vector3f();

        auto normalsAngleCos = ( prj.normal && srcNormals ) ? dot( res.tgtNorm, res.srcNorm ) : 1.0f;

        if ( normalsAngleCos < cosTreshold )
        {
            pairs.active.reset( idx );
            return;
        }
        if ( mutualClosest )
        {
            assert( srcProjector );
            prjObj = res.srcId.objId;
            prj.closestVert = res.srcId.vId;
            // keep prj.distSq
            srcProjector( res.tgtPoint, prj, prjObj );
            if ( prj.closestVert != res.srcId.vId )
                pairs.active.reset( idx );
        }
    } );
}

MultiwayICP::MultiwayICP( const ICPObjects& objects, float samplingVoxelSize ) :
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
    setupLayers_();

    samplingSize_ = samplingVoxelSize;

    Vector<VertBitSet, ObjId> samplesPerObj( objs_.size() );

    ParallelFor( objs_, [&] ( ObjId ind )
    {
        const auto& obj = objs_[ind];
        samplesPerObj[ind] = *obj.obj.pointsGridSampling( samplingVoxelSize );
    } );

    reservePairsLayer0_( std::move( samplesPerObj ) );

    // only do something if cascade mode is required, really should be called on each iteration
    reserveUpperLayerPairs_( resampleUpperLayers_() );
}

float MultiwayICP::getMeanSqDistToPoint() const
{
    NumSum numSum;
    for ( ICPLayer l( 0 ); l < pairsGridPerLayer_.size(); ++l )
    {
        const auto& pairs = pairsGridPerLayer_[l];
        numSum = numSum + tbb::parallel_deterministic_reduce( tbb::blocked_range( size_t( 0 ), pairs.size() * pairs.size() ), NumSum(),
        [&] ( const auto& range, NumSum curr )
        {
            for ( size_t r = range.begin(); r < range.end(); ++r )
            {
                size_t i = r % pairs.size();
                size_t j = r / pairs.size();
                if ( i == j )
                    continue;
                curr = curr + MR::getSumSqDistToPoint( pairs[ICPElementId( i )][ICPElementId( j )] );
            }
            return curr;
        }, [] ( auto a, auto b ) { return a + b; } );
    }
    return numSum.rootMeanSqF();
}

float MultiwayICP::getMeanSqDistToPlane() const
{
    NumSum numSum;
    for ( ICPLayer l( 0 ); l < pairsGridPerLayer_.size(); ++l )
    {
        const auto& pairs = pairsGridPerLayer_[l];
        numSum = numSum + tbb::parallel_deterministic_reduce( tbb::blocked_range( size_t( 0 ), pairs.size() * pairs.size() ), NumSum(),
        [&] ( const auto& range, NumSum curr )
        {
            for ( size_t r = range.begin(); r < range.end(); ++r )
            {
                size_t i = r % pairs.size();
                size_t j = r / pairs.size();
                if ( i == j )
                    continue;
                curr = curr + MR::getSumSqDistToPlane( pairs[ICPElementId( i )][ICPElementId( j )] );
            }
            return curr;
        }, [] ( auto a, auto b ) { return a + b; } );
    }
    return numSum.rootMeanSqF();
}

size_t MultiwayICP::getNumActivePairs() const
{
    size_t num = 0;
    for ( ICPLayer l( 0 ); l < pairsGridPerLayer_.size(); ++l )
    {
        const auto& pairs = pairsGridPerLayer_[l];
        num += tbb::parallel_deterministic_reduce( tbb::blocked_range( size_t( 0 ), pairs.size() * pairs.size() ), size_t( 0 ),
        [&] ( const auto& range, size_t curr )
        {
            for ( size_t r = range.begin(); r < range.end(); ++r )
            {
                size_t i = r % pairs.size();
                size_t j = r / pairs.size();
                if ( i == j )
                    continue;
                curr += MR::getNumActivePairs( pairs[ICPElementId( i )][ICPElementId( j )] );
            }
            return curr;
        }, [] ( auto a, auto b ) { return a + b; } );
    }
    return num;
}

std::string MultiwayICP::getStatusInfo() const
{
    return getICPStatusInfo( iter_, resultType_ );
}

void MultiwayICP::updateAllPointPairs()
{
    MR_TIMER;
    for ( ICPLayer l( 0 ); l < pairsGridPerLayer_.size(); ++l )
        updateLayerPairs_( l );
}

void MultiwayICP::setupLayers_()
{
    if ( maxGroupSize_ <= 1 || objs_.size() <= maxGroupSize_ )
    {
        pairsGridPerLayer_.resize( 1 );
        return;
    }
    int numLayers = 1;
    int numElements = int( objs_.size() );
    while ( numElements > 1 )
    {
        numElements = ( numElements + maxGroupSize_ - 1 ) / maxGroupSize_;
        numLayers++;
    }
    pairsGridPerLayer_.resize( numLayers );
}

void MultiwayICP::reservePairsLayer0_( Vector<VertBitSet, ObjId>&& samplesPerObj )
{
    bool cascadeMode = pairsGridPerLayer_.size() > 1;

    assert( !pairsGridPerLayer_.empty() );
    auto& pairsGrid = pairsGridPerLayer_[0];
    pairsGrid.clear();
    pairsGrid.resize( objs_.size() );
    for ( ICPElementId i( 0 ); i < objs_.size(); ++i )
    {
        auto& pairs = pairsGrid[i];
        pairs.resize( objs_.size() );
        for ( ICPElementId j( 0 ); j < objs_.size(); ++j )
        {
            if ( i == j )
                continue;

            if ( cascadeMode )
            {
                auto groupI = i / maxGroupSize_;
                auto groupJ = j / maxGroupSize_;
                if ( groupI != groupJ )
                    continue;
            }

            auto& thisPairs = pairs[j];
            ObjId srcObj = ObjId( i.get() );
            thisPairs.vec.reserve( samplesPerObj[srcObj].count() );
            for ( auto v : samplesPerObj[srcObj] )
            {
                auto& back = thisPairs.vec.emplace_back();
                back.srcId.objId = srcObj;
                back.srcId.vId = v;
            }
            thisPairs.active.reserve( thisPairs.vec.size() );
            thisPairs.active.clear();
        }
    }
}

MultiwayICP::LayerSamples MultiwayICP::resampleUpperLayers_()
{
    MR_TIMER;
    if ( pairsGridPerLayer_.size() < 2 )
        return {};

    int numLayers = 1;
    int numGroups = ( int( objs_.size() ) + maxGroupSize_ - 1 ) / maxGroupSize_;
    while ( numGroups > 1 )
    {
        numLayers++;
        numGroups = ( numGroups + maxGroupSize_ - 1 ) / maxGroupSize_;
    }

    LayerSamples samples( numLayers );
    int groupSize = 1;
    for ( ICPLayer l = 1; l < numLayers; ++l )
    {
        groupSize *= maxGroupSize_;
        int layerSize = ( int( objs_.size() ) + groupSize - 1 ) / groupSize;
        auto& layerSamples = samples[l];
        layerSamples.resize( layerSize );
        ParallelFor( ICPElementId( 0 ), ICPElementId( layerSize ), [&] ( ICPElementId gId )
        {
            // TODO: use GroupIndexer
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
                smp.objId += first; // works only if Objects in group are continuously stored
        } );
    }
    return samples;
}

void MultiwayICP::reserveUpperLayerPairs_( LayerSamples&& samples )
{
    MR_TIMER;
    if ( samples.empty() )
        return;
    assert( pairsGridPerLayer_.size() > 0 );
    pairsGridPerLayer_.resize( samples.size() );
    for ( ICPLayer l( 1 ); l < pairsGridPerLayer_.size(); ++l )
    {
        const auto& samplesOnLayer = samples[l];
        auto& pairsOnLayer = pairsGridPerLayer_[l];
        int numGroups = int( samplesOnLayer.size() );
        pairsOnLayer.resize( numGroups );
        for ( ICPElementId gI( 0 ); gI < numGroups; ++gI )
        {
            auto& pairsOnGroup = pairsOnLayer[gI];
            pairsOnGroup.resize( numGroups );
            const auto& thisSamples = samplesOnLayer[gI];
            for ( ICPElementId gJ( 0 ); gJ < numGroups; ++gJ )
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

void MultiwayICP::updateLayerPairs_( ICPLayer l )
{
    MR_TIMER;
    int groupSize = 1;
    for ( int i = 1; i <= l; ++i )
        groupSize *= maxGroupSize_;
    auto numGroups = pairsGridPerLayer_[l].size();
    bool cascadeMode = pairsGridPerLayer_.size() > 1;

    assert( l == 0 || cascadeMode );
    assert( l < pairsGridPerLayer_.size() );

    // build trees
    using LayerTrees = Vector<AABBTreeObjects, ICPElementId>;
    LayerTrees trees;
    if ( l > 0 )
    {
        // only for upper layers
        trees.resize( numGroups );
        ParallelFor( trees, [&] ( ICPElementId gI )
        {
            // TODO: use GroupIndexer
            auto gFirst = ObjId( gI * groupSize );
            auto gLast = ObjId( std::min( ( gI + 1 ) * groupSize, int( objs_.size() ) ) );

            ICPObjects gObjs( begin( objs_ ) + gFirst, begin( objs_ ) + gLast );
            trees[gI] = AABBTreeObjects( std::move( gObjs ) );
        } );
    }

    auto createProjector = [&] ( ICPElementId elId, ObjId first )->ICPGroupProjector
    {
        if ( l == 0 )
            return [this, elId] ( const Vector3f& p, MeshOrPoints::ProjectionResult& res, ObjId& resId ) mutable
        {
            auto proj = objs_[ObjId( elId.get() )].obj.limitedProjector();
            proj( objs_[ObjId( elId.get() )].xf.inverse()( p ), res );
            if ( res.closestVert )
                resId = ObjId( elId.get() );
        };
        return [&trees, elId, first] ( const Vector3f& p, MeshOrPoints::ProjectionResult& res, ObjId& resId )
        {
            projectOnAll( p, trees[elId], res.distSq, [&] ( ObjId oId, MeshOrPoints::ProjectionResult prj )
            {
                if ( prj.distSq >= res.distSq )
                    return;
                res = prj;
                resId = oId + first;
            } );
        };
    };


    // update pairs
    auto gridSize = numGroups * numGroups;
    ParallelFor( 0, int( gridSize ), [&] ( int gridId )
    {
        ICPElementId gI = ICPElementId( gridId / numGroups );
        ICPElementId gJ = ICPElementId( gridId % numGroups );
        if ( gI == gJ )
            return;

        if ( cascadeMode )
        {
            auto hyperGroupI = gI / maxGroupSize_;
            auto hyperGroupJ = gJ / maxGroupSize_;
            if ( hyperGroupI != hyperGroupJ )
                return;
        }

        // TODO: use GroupIndexer
        auto srcFirst = ObjId( gI * groupSize );
        auto tgtFirst = ObjId( gJ * groupSize );

        MR::updateGroupPairs( pairsGridPerLayer_[l][gI][gJ], objs_, createProjector( gI, srcFirst ), createProjector( gJ, tgtFirst ), prop_.cosTreshold, prop_.distThresholdSq, prop_.mutualClosest );
    } );
    deactivateFarDistPairs_( l );
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

void MultiwayICP::deactivateFarDistPairs_( ICPLayer l )
{
    MR_TIMER;
    auto& pairs = pairsGridPerLayer_[l];
    Vector<float, ICPElementId> maxDistSq( pairs.size() );
    for ( int it = 0; it < 3; ++it )
    {
        ParallelFor( maxDistSq, [&] ( auto id )
        {
            maxDistSq[id] = sqr( prop_.farDistFactor * MR::calcMeanSqDistToPointEx( pairs, id ) );
        } );

        tbb::enumerable_thread_specific<size_t> counters( 0 );
        ParallelFor( size_t( 0 ), pairs.size() * pairs.size(), [&] ( size_t r )
        {
            auto i = ICPElementId( r % pairs.size() );
            auto j = ICPElementId( r / pairs.size() );
            if ( i == j )
                return;
            if ( maxDistSq[i] >= prop_.distThresholdSq )
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

bool MultiwayICP::doIteration_( bool p2pl )
{
    if ( pairsGridPerLayer_.size() > 1 )
        return cascadeIter_( p2pl );

    updateAllPointPairs();

    if ( maxGroupSize_ == 1 )
        return p2pl ? p2plIter_() : p2ptIter_();

    return multiwayIter_( p2pl );
}

bool MultiwayICP::p2ptIter_()
{
    MR_TIMER;
    using FullSizeBool = uint8_t;
    Vector<FullSizeBool, ObjId> valid( objs_.size() );
    ParallelFor( objs_, [&] ( ObjId objId )
    {
        ICPElementId id( objId.get() );
        const auto& pairs = pairsGridPerLayer_[0];
        PointToPointAligningTransform p2pt;
        for ( ICPElementId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairs[id][j].active )
            {
                const auto& vp = pairs[id][j].vec[idx];
                p2pt.add( vp.srcPoint, 0.5f * ( vp.srcPoint + vp.tgtPoint ), vp.weight );
            }
            for ( size_t idx : pairs[j][id].active )
            {
                const auto& vp = pairs[j][id].vec[idx];
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
            valid[objId] = FullSizeBool( false );
            return;
        }
        objs_[objId].xf = res * objs_[objId].xf;
        valid[objId] = FullSizeBool( true );
    } );

    return std::all_of( valid.vec_.begin(), valid.vec_.end(), [] ( auto v ) { return bool( v ); } );
}

bool MultiwayICP::p2plIter_()
{
    MR_TIMER;
    using FullSizeBool = uint8_t;
    Vector<FullSizeBool, ObjId> valid( objs_.size() );
    ParallelFor( objs_, [&] ( ObjId objId )
    {
        ICPElementId id( objId.get() );
        const auto& pairs = pairsGridPerLayer_[0];
        Vector3f centroidRef;
        int activeCount = 0;
        for ( ICPElementId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairs[id][j].active )
            {
                const auto& vp = pairs[id][j].vec[idx];
                centroidRef += ( vp.tgtPoint + vp.srcPoint ) * 0.5f;
                centroidRef += vp.srcPoint;
                ++activeCount;
            }
            for ( size_t idx : pairs[j][id].active )
            {
                const auto& vp = pairs[j][id].vec[idx];
                centroidRef += ( vp.tgtPoint + vp.srcPoint ) * 0.5f;
                centroidRef += vp.tgtPoint;
                ++activeCount;
            }
        }
        if ( activeCount <= 0 )
        {
            valid[objId] = FullSizeBool( false );
            return;
        }
        centroidRef /= float( activeCount * 2 );
        AffineXf3f centroidRefXf = AffineXf3f( Matrix3f(), centroidRef );

        PointToPlaneAligningTransform p2pl;
        for ( ICPElementId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == id )
                continue;
            for ( size_t idx : pairs[id][j].active )
            {
                const auto& vp = pairs[id][j].vec[idx];
                p2pl.add( vp.srcPoint - centroidRef, ( vp.tgtPoint + vp.srcPoint ) * 0.5f - centroidRef, vp.tgtNorm, vp.weight );
            }
            for ( size_t idx : pairs[j][id].active )
            {
                const auto& vp = pairs[j][id].vec[idx];
                p2pl.add( vp.tgtPoint - centroidRef, ( vp.tgtPoint + vp.srcPoint ) * 0.5f - centroidRef, vp.srcNorm, vp.weight );
            }
        }
        p2pl.prepare();

        AffineXf3f res = getAligningXf( p2pl, prop_.icpMode, prop_.p2plAngleLimit, prop_.p2plScaleLimit, prop_.fixedRotationAxis );
        if ( std::isnan( res.b.x ) ) //nan check
        {
            valid[objId] = FullSizeBool( false );
            return;
        }
        objs_[objId].xf = centroidRefXf * res * centroidRefXf.inverse() * objs_[objId].xf;
        valid[objId] = FullSizeBool( true );
    } );
    return std::all_of( valid.vec_.begin(), valid.vec_.end(), [] ( auto v ) { return bool( v ); } );
}

bool MultiwayICP::multiwayIter_( bool p2pl )
{
    MR_TIMER;
    Vector<MultiwayAligningTransform,ObjId> mats( objs_.size() );
    ParallelFor( mats, [&] ( ObjId objId )
    {
        ICPElementId i( objId.get() );
        const auto& pairs = pairsGridPerLayer_[0];
        auto& mat = mats[objId];
        mat.reset( int( objs_.size() ) );
        for ( ICPElementId j( 0 ); j < objs_.size(); ++j )
        {
            if ( j == i )
                continue;
            for ( auto idx : pairs[i][j].active )
            {
                const auto& data = pairs[i][j].vec[idx];
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

bool MultiwayICP::cascadeIter_( bool p2pl /*= true */ )
{
    int elemetSize = 1;
    for ( ICPLayer l = 0; l < pairsGridPerLayer_.size(); ++l )
    {
        updateLayerPairs_( l );

        const auto& pairsOnLayer = pairsGridPerLayer_[l];
        int numHyperGroups = ( int( pairsOnLayer.size() ) + maxGroupSize_ - 1 ) / maxGroupSize_;

        for ( int hgI = 0; hgI < numHyperGroups; ++hgI )
        {
            int numElements = hgI + 1 == numHyperGroups ? int( pairsOnLayer.size() ) - hgI * maxGroupSize_ : maxGroupSize_;
            assert( numElements > 0 );
            if ( numElements == 1 )
                continue;

            MultiwayAligningTransform mat;
            mat.reset( numElements );
            for ( int localElI( 0 ); localElI < numElements; ++localElI )
            for ( int localElJ( 0 ); localElJ < numElements; ++localElJ )
            {
                if ( localElI == localElJ )
                    continue;
                auto elI = ICPElementId( hgI * maxGroupSize_ + localElI );
                auto elJ = ICPElementId( hgI * maxGroupSize_ + localElJ );
                const auto& pairs = pairsGridPerLayer_[l][elI][elJ];
                for ( auto idx : pairs.active )
                {
                    const auto& data = pairs.vec[idx];
                    if ( p2pl )
                        mat.add( localElI, data.srcPoint, localElJ, data.tgtPoint, ( data.tgtNorm + data.srcNorm ).normalized(), data.weight );
                    else
                        mat.add( localElI, data.srcPoint, localElJ, data.tgtPoint, data.weight );
                }
            }

            MultiwayAligningTransform::Stabilizer stabilizer;
            stabilizer.rot = samplingSize_ * 1e-1f;
            stabilizer.shift = 1e-3f;
            auto res = mat.solve( stabilizer );
            for ( int localElI( 0 ); localElI < numElements; ++localElI )
            {
                auto resI = res[localElI].rigidXf();
                if ( std::isnan( resI.b.x ) )
                    return false;
                // TODO: use GroupIndexer
                ObjId iBegin = ObjId( hgI * maxGroupSize_ * elemetSize + localElI * elemetSize );
                ObjId iEnd = std::min( iBegin + elemetSize, ObjId( objs_.size() ) );
                for ( ObjId i( iBegin ); i < iEnd; ++i )
                    objs_[i].xf = AffineXf3f( resI * AffineXf3d( objs_[i].xf ) );
            }
        }
        elemetSize *= maxGroupSize_;
    }
    return true;
}

}
