#include "MRMeshDecimate.h"
#include "MRMesh.h"
#include "MRQuadraticForm.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRTriMath.h"
#include "MRTimer.h"
#include "MRMeshDelone.h"
#include "MRMeshSubdivide.h"
#include "MRMeshRelax.h"
#include "MRLineSegm.h"
#include "MRPriorityQueue.h"
#include "MRBuffer.h"
#include "MRTbbThreadMutex.h"
#include "MRMeshFixer.h"
#include "MRObjectMeshData.h"
#include "MRMeshAttributesToUpdate.h"
#include "MRMeshDecimateCallbacks.h"
#include "MRMapEdge.h"
#include "MRObjectMesh.h"
#include "MRAABBTree.h"

namespace MR
{

namespace
{

enum class EdgeOp : unsigned int
{
    CollapseOptPos, ///< collapse the edge with target position optimization
    CollapseEnd,    ///< collapse the edge in one of its current vertices
    Flip            ///< flip the edge inside quadrangle
    // one more option is available to fit in 2 bits
};

struct QueueElement
{
    float c = 0;
    struct X {
        EdgeOp edgeOp : 2 = EdgeOp::CollapseOptPos;
        unsigned int uedgeId : 30 = 0;
    } x;
    UndirectedEdgeId uedgeId() const { return UndirectedEdgeId{ (int)x.uedgeId }; }
    std::pair<float, int> asPair() const { return { -c, x.uedgeId }; }
    bool operator < ( const QueueElement & r ) const { return asPair() < r.asPair(); }
};
static_assert( sizeof( QueueElement ) == 8 );

} // anonymous namespace

class MeshDecimator
{
public:
    /// if initialEdges is given, then only these edges are put in the queue initially, and other region edges only after nearby changes
    MeshDecimator( Mesh & mesh, const DecimateSettings & settings, const UndirectedEdgeBitSet * initialEdges = nullptr );
    DecimateResult run();

private:
    Mesh & mesh_;
    const DecimateSettings & settings_;
    const DeloneSettings deloneSettings_;
    const float maxErrorSq_;
    Vector<QuadraticForm3f, VertId> myVertForms_;
    Vector<QuadraticForm3f, VertId> * pVertForms_ = nullptr;
    UndirectedEdgeBitSet regionEdges_;
    VertBitSet myBdVerts_;
    const VertBitSet * pBdVerts_ = nullptr;
    const UndirectedEdgeBitSet * initialEdges_ = nullptr;
    std::function<void( EdgeId del, EdgeId rem )> onEdgeDel_;

    PriorityQueue<QueueElement> queue_;
    UndirectedEdgeBitSet validInQueue_; // bit set if the edge is both present in queue_ and not lone
    UndirectedEdgeBitSet outdated_; // true if edge's error in the queue may be outdated (too optimistic) due to nearby collapse
    size_t numOutdated_ = 0; // total number of not lone outdated edges in the queue
    size_t maxRemainingFlips_ = 0; // the number of flip-edge operations that can be performed before stop adding them in the queue
    DecimateResult res_;
    Vector<VertId> originNeis_;
    Vector<Vector3f> triDblAreas_; // directed double areas of newly formed triangles to check that they are consistently oriented
    class EdgeMetricCalc;

    bool initialize_();
    void initializeQueue_();
    Vector<QueueElement> makeQueueElements_();
    void updateQueue_();
    QuadraticForm3f collapseForm_( UndirectedEdgeId ue, const Vector3f & collapsePos ) const;
    std::optional<QueueElement> computeQueueElement_( UndirectedEdgeId ue, bool optimizeVertexPos,
        QuadraticForm3f * outCollapseForm = nullptr, Vector3f * outCollapsePos = nullptr ) const;

    /// adds given edge in the queue if it was missing there;
    /// returns true only if the edge was already in queue with probably outdated error
    bool addInQueueIfMissing_( UndirectedEdgeId ue );

    /// adds given edges (currently missing) in queue
    void addInQueue_( UndirectedEdgeId ue, bool optimizeVertexPos );

    void flipEdge_( UndirectedEdgeId ue );

    enum class CollapseStatus
    {
        Ok,          ///< collapse is possible or was successful
        SharedEdge,  ///< cannot collapse internal edge if its left and right faces share another edge
        PosFarBd,    ///< new vertex position is too far from collapsing boundary edge
        MultipleEdge,///< cannot collapse an edge because there is another edge in mesh with same ends
        Flippable,   ///< cannot collapse a flippable edge incident to a not-flippable edge
        TouchBd,     ///< prohibit collapse of an inner edge if it brings two boundaries in touch
        TriAspect,   ///< new triangle aspect ratio would be larger than all of old triangle aspect ratios and larger than allowed in settings
        LongEdge,    ///< new edge would be longer than all of old edges and longer than allowed in settings
        NormalFlip,  ///< if at least one triangle normal flips, checks that all new normals are consistent (do not check for degenerate edges)
        User         ///< user callback prohibits the collapse
    };

    /// true if collapse failed due to a geometric criterion, and may succeed for another collapse position
    static bool geomFail_( CollapseStatus s )
    {
        return
            s == CollapseStatus::TriAspect || s == CollapseStatus::NormalFlip ||
            s == CollapseStatus::PosFarBd || s == CollapseStatus::LongEdge;
    }

    struct CanCollapseRes
    {
        /// if collapse cannot be done then it is invalid,
        /// otherwise it is input edge, oriented such that org( e ) will be the remaining vertex after collapse
        EdgeId e;
        CollapseStatus status = CollapseStatus::Ok;
    };
    CanCollapseRes canCollapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos ); // not const because it changes temporary originNeis_ and triDblAreas_

    /// performs edge collapse after previous successful check by canCollapse_,
    /// if one of the edge's vertices remain, then stores (collapseForm) as its new form and updates error of its neighbor edges
    /// \return org( edgeToCollapse ) or invalid id if it was the last edge
    VertId forceCollapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos, const QuadraticForm3f & collapseForm );

    struct CollapseRes
    {
        /// if collapse failed (or it was the last edge) then it is invalid, otherwise it is the remaining vertex
        VertId v;
        CollapseStatus status = CollapseStatus::Ok;
    };
    /// canCollapse_ + forceCollapse_
    CollapseRes collapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos, const QuadraticForm3f & collapseForm );
};

MeshDecimator::MeshDecimator( Mesh & mesh, const DecimateSettings & settings, const UndirectedEdgeBitSet * initialEdges )
    : mesh_( mesh )
    , settings_( settings )
    , deloneSettings_{
        .maxAngleChange = settings.maxAngleChange,
        .criticalTriAspectRatio = settings.criticalTriAspectRatio,
        .region = settings.region }
    , maxErrorSq_( sqr( settings.maxError ) )
    , initialEdges_( initialEdges )
{
    onEdgeDel_ =
    [
        this,
        notFlippable = settings_.notFlippable,
        edgesToCollapse = settings_.edgesToCollapse,
        twinMap = settings_.twinMap,
        onEdgeDel = settings_.onEdgeDel
    ] ( EdgeId del, EdgeId rem )
    {
        validInQueue_.reset( del );
        if ( outdated_.test_set( del, false ) )
            --numOutdated_;

        if ( notFlippable && notFlippable->test_set( del.undirected(), false ) && rem )
            notFlippable->autoResizeSet( rem.undirected() );

        if ( edgesToCollapse && edgesToCollapse->test_set( del.undirected(), false ) && rem )
            edgesToCollapse->autoResizeSet( rem.undirected() );

        if ( twinMap )
        {
            auto itDel = twinMap->find( del );
            if ( itDel != twinMap->end() )
            {
                auto tgt = itDel->second;
                auto itTgt = twinMap->find( tgt );
                assert( itTgt != twinMap->end() );
                assert( itTgt->second == del.undirected() );
                twinMap->erase( itDel );
                if ( rem )
                {
                    assert( twinMap->count( rem ) == 0 );
                    (*twinMap)[rem] = tgt;
                    itTgt->second = rem;
                }
                else
                    twinMap->erase( itTgt );
            }
        }

        if ( onEdgeDel )
            onEdgeDel( del, rem );
    };
}

QuadraticForm3f computeFormAtVertex( const MR::MeshPart & mp, MR::VertId v, float stabilizer, bool angleWeigted, const UndirectedEdgeBitSet * creases )
{
    QuadraticForm3f qf = mp.mesh.quadraticForm( v, angleWeigted, mp.region, creases );
    qf.addDistToOrigin( stabilizer );
    return qf;
}

Vector<QuadraticForm3f, VertId> computeFormsAtVertices( const MeshPart & mp, float stabilizer, bool angleWeigted, const UndirectedEdgeBitSet * creases )
{
    MR_TIMER;

    VertBitSet store;
    const VertBitSet & regionVertices = getIncidentVerts( mp.mesh.topology, mp.region, store );

    Vector<QuadraticForm3f, VertId> res( regionVertices.find_last() + 1 );
    BitSetParallelFor( regionVertices, [&]( VertId v )
    {
        res[v] = computeFormAtVertex( mp, v, stabilizer, angleWeigted, creases );
    } );

    return res;
}

bool resolveMeshDegenerations( Mesh & mesh, const ResolveMeshDegenSettings & settings )
{
    MR_TIMER;

    DecimateSettings dsettings
    {
        .maxError = settings.maxDeviation,
        .criticalTriAspectRatio = settings.criticalAspectRatio,
        .tinyEdgeLength = settings.tinyEdgeLength,
        .stabilizer = settings.stabilizer,
        .optimizeVertexPos = false, // this decreases probability of normal inversion near mesh degenerations
        .region = settings.region,
        .maxAngleChange = settings.maxAngleChange
    };
    return decimateMesh( mesh, dsettings ).vertsDeleted > 0;
}

bool MeshDecimator::initialize_()
{
    MR_TIMER;

    if ( settings_.vertForms )
        pVertForms_ = settings_.vertForms;
    else
        pVertForms_ = &myVertForms_;

    if ( pVertForms_->empty() )
        *pVertForms_ = computeFormsAtVertices( MeshPart{ mesh_, settings_.region }, settings_.stabilizer, settings_.angleWeightedDistToPlane, settings_.notFlippable );

    if ( settings_.progressCallback && !settings_.progressCallback( 0.1f ) )
        return false;

    // initialize regionEdges_ if some edges (out-of-region or touching boundary) cannot be collapsed
    if ( settings_.region )
    {
        // all region edges
        regionEdges_ = settings_.regionEdges ? *settings_.regionEdges : getIncidentEdges( mesh_.topology, *settings_.region );
        if ( settings_.edgesToCollapse )
            regionEdges_ &= *settings_.edgesToCollapse;
        if ( !settings_.touchNearBdEdges )
        {
            // exclude edges touching boundary
            BitSetParallelFor( regionEdges_, [&]( UndirectedEdgeId ue )
            {
                if ( pBdVerts_->test( mesh_.topology.org( ue ) ) ||
                     pBdVerts_->test( mesh_.topology.dest( ue ) ) )
                    regionEdges_.reset( ue );
            } );
        }
    }
    else if ( !settings_.touchNearBdEdges )
    {
        assert( !settings_.region );
        regionEdges_.clear();
        regionEdges_.resize( mesh_.topology.undirectedEdgeSize(), true );
        // exclude lone edges and edges touching boundary
        BitSetParallelForAll( regionEdges_, [&]( UndirectedEdgeId ue )
        {
            if ( ( settings_.edgesToCollapse && !settings_.edgesToCollapse->test( ue ) ) ||
                 mesh_.topology.isLoneEdge( ue ) ||
                 pBdVerts_->test( mesh_.topology.org( ue ) ) ||
                 pBdVerts_->test( mesh_.topology.dest( ue ) ) )
                regionEdges_.reset( ue );
        } );
    }
    else if ( settings_.edgesToCollapse )
        regionEdges_ = *settings_.edgesToCollapse;

    if ( settings_.progressCallback && !settings_.progressCallback( 0.15f ) )
        return false;

    // (maxRemainingFlips_ = 1) allows adding flip operations in the queue
    maxRemainingFlips_ = settings_.maxAngleChange >= 0 ? 1 : 0;
    initializeQueue_();

    if ( settings_.progressCallback && !settings_.progressCallback( 0.25f ) )
        return false;
    return true;
}

auto MeshDecimator::makeQueueElements_() -> Vector<QueueElement>
{
    MR_TIMER;

    const auto sz = mesh_.topology.undirectedEdgeSize();
    validInQueue_.clear();
    validInQueue_.resize( sz, false );

    tbb::enumerable_thread_specific<Vector<QueueElement>> threadData;
    if ( initialEdges_ )
    {
        // few initial edges in a large bit set
        Vector<UndirectedEdgeId> edges;
        for ( auto ue : *initialEdges_ )
            if ( regionEdges_.empty() ? !mesh_.topology.isLoneEdge( ue ) : regionEdges_.test( ue ) )
                edges.push_back( ue );
        ParallelFor( edges, [&]( size_t i )
        {
            if ( auto qe = computeQueueElement_( edges[i], settings_.optimizeVertexPos ) )
                threadData.local().push_back( *qe );
        } );
        for ( const auto & v : threadData )
            for ( const auto & qe : v )
                validInQueue_.set( qe.uedgeId() );
    }
    else
    {
        BitSetParallelForAll( validInQueue_, [&]( UndirectedEdgeId ue )
        {
            if ( regionEdges_.empty() )
            {
                if ( mesh_.topology.isLoneEdge( ue ) )
                    return;
            }
            else
            {
                if ( !regionEdges_.test( ue ) )
                    return;
            }
            if ( auto qe = computeQueueElement_( ue, settings_.optimizeVertexPos ) )
            {
                threadData.local().push_back( *qe );
                validInQueue_.set( ue );
            }
        } );
    }

    size_t queueSize = 0;
    for ( const auto & v : threadData )
        queueSize += v.size();

    Vector<QueueElement> elms;
    elms.reserve( queueSize );
    for ( auto & v : threadData )
    {
        elms.insert( elms.end(), v.begin(), v.end() );
        v = {};
    }
    assert( elms.size() == queueSize );
    // prevents never ending flips scenario, (maxRemainingFlips_ = queueSize) was not enough in real cases
    maxRemainingFlips_ = settings_.maxAngleChange >= 0 ? 10 * queueSize : 0;
    return elms;
}

void MeshDecimator::initializeQueue_()
{
    MR_TIMER;

    // free space occupied by existing queue
    queue_ = {};

    queue_ = PriorityQueue<QueueElement>{ std::less<QueueElement>(), makeQueueElements_() };

    outdated_.clear();
    outdated_.resize( mesh_.topology.undirectedEdgeSize(), false );
    numOutdated_ = 0;
}

void MeshDecimator::updateQueue_()
{
    MR_TIMER;

    Timer t( "compute" );
    auto & vec = queue_.c;
    // recompute errors for outdated edges
    BitSet del( vec.size(), false );
    BitSetParallelForAll( del, [&]( size_t i )
    {
        auto& qe = vec[i];
        const auto ue = qe.uedgeId();
        if ( !validInQueue_.test( ue ) )
            return;
        if ( !outdated_.test( ue ) )
            return;
        if ( auto n = computeQueueElement_( qe.uedgeId(), qe.x.edgeOp == EdgeOp::CollapseOptPos ) )
            qe = *n;
        else
            del.set( i );
    } );
    outdated_.reset( 0_ue, outdated_.size() );
    numOutdated_ = 0;

    t.restart( "invalidate" );
    // removed valid flag for outdated edges with failed computeQueueElement_
    for ( auto i : del )
        validInQueue_.reset( vec[i].uedgeId() );

    t.restart( "remove deleted" );
    // remove invalid and deleted edges from the queue
    erase_if( vec, [&]( QueueElement & qe ) { return !validInQueue_.test( qe.uedgeId() ); } );

    t.restart( "restore heap" );
    // sort elements to restore heap property
    std::make_heap( vec.begin(), vec.end() );
}

QuadraticForm3f MeshDecimator::collapseForm_( UndirectedEdgeId ue, const Vector3f & collapsePos ) const
{
    EdgeId e{ ue };
    const auto o = mesh_.topology.org( e );
    const auto d = mesh_.topology.dest( e );
    const auto po = mesh_.points[o];
    const auto pd = mesh_.points[d];
    const auto vo = (*pVertForms_)[o];
    const auto vd = (*pVertForms_)[d];
    return sumAt( vo, po, vd, pd, collapsePos );
}

auto MeshDecimator::computeQueueElement_( UndirectedEdgeId ue, bool optimizeVertexPos,
    QuadraticForm3f * outCollapseForm, Vector3f * outCollapsePos ) const -> std::optional<QueueElement>
{
    EdgeId e{ ue };
    const auto o = mesh_.topology.org( e );
    const auto d = mesh_.topology.dest( e );
    const auto po = mesh_.points[o];
    const auto pd = mesh_.points[d];
    const auto vo = (*pVertForms_)[o];
    const auto vd = (*pVertForms_)[d];

    std::optional<QueueElement> res;
    // prepares res; checks flip metric; returns true if the edge does not collpase and function can return
    auto earlyReturn = [&]( float errSq )
    {
        EdgeOp edgeOp = optimizeVertexPos ? EdgeOp::CollapseOptPos : EdgeOp::CollapseEnd;
        if ( maxRemainingFlips_ > 0 && ( !settings_.notFlippable || !settings_.notFlippable->test( ue ) ) )
        {
            float deviationSqAfterFlip = FLT_MAX;
            if ( !checkDeloneQuadrangleInMesh( mesh_, ue, deloneSettings_, &deviationSqAfterFlip )
                && deviationSqAfterFlip < errSq )
            {
                edgeOp = EdgeOp::Flip;
                errSq = deviationSqAfterFlip;
            }
        }
        if ( ( edgeOp == EdgeOp::Flip || !settings_.adjustCollapse ) && errSq > maxErrorSq_ )
            return true;
        res.emplace();
        res->x.uedgeId = (int)ue;
        res->x.edgeOp = edgeOp;
        res->c = errSq;
        return edgeOp == EdgeOp::Flip;
    };

    if ( settings_.strategy == DecimateStrategy::ShortestEdgeFirst )
    {
        if ( earlyReturn( mesh_.edgeLengthSq( e ) ) )
            return res;
    }

    QuadraticForm3f qf;
    Vector3f pos;
    if ( settings_.touchBdVerts // if boundary vertices can be moved ...
        || ( settings_.notFlippable && settings_.notFlippable->test( ue ) ) ) // ... or this is a not-flippable edge (if you do not want such collapses then exclude it from settings_.edgesToCollapse)
    {
        std::tie( qf, pos ) = sum( vo, po, vd, pd, !optimizeVertexPos );
    }
    else
    {
        const bool bdO = pBdVerts_->test( o );
        const bool bdD = pBdVerts_->test( d );
        if ( bdO )
        {
            if ( bdD )
                qf.c = FLT_MAX;
            else
                qf = sumAt( vo, po, vd, pd, pos = po );
        }
        else if ( bdD )
            qf = sumAt( vo, po, vd, pd, pos = pd );
        else
            std::tie( qf, pos ) = sum( vo, po, vd, pd, !optimizeVertexPos );
    }

    if ( settings_.strategy == DecimateStrategy::MinimizeError )
    {
        if ( earlyReturn( qf.c ) )
            return res;
    }

    assert( res && res->x.edgeOp != EdgeOp::Flip );
    if ( settings_.adjustCollapse )
    {
        const auto pos0 = pos;
        settings_.adjustCollapse( ue, res->c, pos );
        if ( res->c > maxErrorSq_ )
            return {};
        if ( outCollapseForm && pos != pos0 )
            qf.c = vo.eval( po - pos ) + vd.eval( pd - pos );
    }

    if ( outCollapseForm )
        *outCollapseForm = qf;
    if ( outCollapsePos )
        *outCollapsePos = pos;

    return res;
}

bool MeshDecimator::addInQueueIfMissing_( UndirectedEdgeId ue )
{
    if ( !regionEdges_.empty() && !regionEdges_.test( ue ) )
        return false;
    if ( validInQueue_.test( ue ) )
        return true;
    addInQueue_( ue, settings_.optimizeVertexPos );
    return false;
}

void MeshDecimator::addInQueue_( UndirectedEdgeId ue, bool optimizeVertexPos )
{
    assert ( !validInQueue_.test( ue ) );
    if ( auto qe = computeQueueElement_( ue, optimizeVertexPos ) )
    {
        queue_.push( *qe );
        validInQueue_.set( ue );
        if ( outdated_.test_set( ue, false ) )
            --numOutdated_;
    }
}

void MeshDecimator::flipEdge_( UndirectedEdgeId ue )
{
    EdgeId e = ue;
    mesh_.topology.flipEdge( e );
    assert( mesh_.topology.left( e ) );
    assert( mesh_.topology.right( e ) );
    addInQueueIfMissing_( e.undirected() );
    for ( auto oe : orgRing0( mesh_.topology, e ) )
        addInQueueIfMissing_( oe.undirected() );
    for ( auto oe : orgRing0( mesh_.topology, e.sym() ) )
        addInQueueIfMissing_( oe.undirected() );
    for ( auto oe : orgRing0( mesh_.topology, mesh_.topology.next( e ).sym() ) )
        addInQueueIfMissing_( oe.undirected() );
    for ( auto oe : orgRing0( mesh_.topology, mesh_.topology.prev( e ).sym() ) )
        addInQueueIfMissing_( oe.undirected() );

    if ( maxRemainingFlips_ > 0 )
    {
        --maxRemainingFlips_;
        assert( maxRemainingFlips_ > 0 ); // otherwise either the limit on the number of flips is too tight or flips are in never ending cycle
    }
}

auto MeshDecimator::canCollapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos ) -> CanCollapseRes
{
    const auto & topology = mesh_.topology;
    auto vl = topology.left( edgeToCollapse ).valid()  ? topology.dest( topology.next( edgeToCollapse ) ) : VertId{};
    auto vr = topology.right( edgeToCollapse ).valid() ? topology.dest( topology.prev( edgeToCollapse ) ) : VertId{};

    // cannot collapse internal edge if its left and right faces share another edge
    if ( vl && vr )
    {
        bool oDegree2 = false;
        if ( auto pe = topology.prev( edgeToCollapse ); pe != edgeToCollapse && pe == topology.next( edgeToCollapse ) )
            oDegree2 = true; // (pe) is shared between left and right faces of edgeToCollapse

        bool dDegree2 = false;
        if ( auto pe = topology.prev( edgeToCollapse.sym() ); pe != edgeToCollapse.sym() && pe == topology.next( edgeToCollapse.sym() ) )
            dDegree2 = true; // (pe) is shared between left and right faces of edgeToCollapse
        
        // if both oDegree2 and dDegree2 are true, then vl == vr
        assert( !oDegree2 || !dDegree2 || vl == vr );

        // but can collapse if left and right faces of edgeToCollapse share all 3 edges
        if ( oDegree2 != dDegree2 )
            return { .status =  CollapseStatus::SharedEdge };
    }
    const bool collapsingFlippable = !settings_.notFlippable || !settings_.notFlippable->test( edgeToCollapse );

    auto vo = topology.org( edgeToCollapse );
    auto vd = topology.dest( edgeToCollapse );
    auto po = mesh_.points[vo];
    auto pd = mesh_.points[vd];
    if ( collapsePos == pd )
    {
        // reverse the edge to have its origin in remaining fixed vertex
        edgeToCollapse = edgeToCollapse.sym();
        std::swap( vo, vd );
        std::swap( vl, vr );
        std::swap( po, pd );
    }

    auto smallShift = [maxBdShiftSq = sqr( settings_.maxBdShift )]( const LineSegm3f & segm, const Vector3f & p )
    {
        return ( closestPointOnLineSegm( p, segm ) - p ).lengthSq() <= maxBdShiftSq;
    };
    if ( ( !vl || !vr ) && settings_.maxBdShift < FLT_MAX )
    {
        if ( !smallShift( mesh_.edgeSegment( edgeToCollapse ), collapsePos ) )
            return { .status =  CollapseStatus::PosFarBd }; // new vertex is too far from collapsing boundary edge
        if ( !vr )
        {
            if ( !smallShift( LineSegm3f{ mesh_.orgPnt( mesh_.topology.prevLeftBd( edgeToCollapse ) ), collapsePos }, po ) )
                return { .status =  CollapseStatus::PosFarBd }; // origin of collapsing boundary edge is too far from new boundary segment
            if ( !smallShift( LineSegm3f{ mesh_.destPnt( mesh_.topology.nextLeftBd( edgeToCollapse ) ), collapsePos }, pd ) )
                return { .status =  CollapseStatus::PosFarBd }; // destination of collapsing boundary edge is too far from new boundary segment
        }
        if ( !vl )
        {
            if ( !smallShift( LineSegm3f{ mesh_.orgPnt( mesh_.topology.prevLeftBd( edgeToCollapse.sym() ) ), collapsePos }, pd ) )
                return { .status =  CollapseStatus::PosFarBd }; // destination of collapsing boundary edge is too far from new boundary segment
            if ( !smallShift( LineSegm3f{ mesh_.destPnt( mesh_.topology.nextLeftBd( edgeToCollapse.sym() ) ), collapsePos }, po ) )
                return { .status =  CollapseStatus::PosFarBd }; // origin of collapsing boundary edge is too far from new boundary segment
        }
    }

    // triangle aspect ratios are needed for the aspect ratio check and to find not-degenerate triangles for the normal flip check
    const bool checkAspectRatios = settings_.preventNormalFlips || settings_.maxTriangleAspectRatio < FLT_MAX;
    float maxOldAspectRatio = settings_.maxTriangleAspectRatio;
    float maxNewAspectRatio = 0;
    const float edgeLenSq = ( po - pd ).lengthSq();
    float maxOldEdgeLenSq = std::max( sqr( settings_.maxEdgeLen ), edgeLenSq );
    float maxNewEdgeLenSq = 0;

    originNeis_.clear();
    triDblAreas_.clear(); // new directed areas of triangles that flip their normal or became not-degenerate from degenerate
    Vector3d sumDblArea;
    EdgeId oBdEdge; // a boundary edge !right(e) incident to org( edgeToCollapse )
    for ( EdgeId e : orgRing0( topology, edgeToCollapse ) )
    {
        const auto eDest = topology.dest( e );
        if ( eDest == vd )
            return { .status =  CollapseStatus::MultipleEdge }; // multiple edge found
        if ( eDest != vl && eDest != vr )
            originNeis_.push_back( eDest );

        const auto pDest = mesh_.points[eDest];
        maxOldEdgeLenSq = std::max( maxOldEdgeLenSq, ( po - pDest ).lengthSq() );
        maxNewEdgeLenSq = std::max( maxNewEdgeLenSq, ( collapsePos - pDest ).lengthSq() );
        if ( !topology.left( e ) )
        {
            oBdEdge = topology.next( e );
            assert( topology.isLeftBdEdge( oBdEdge ) );
            continue;
        }
        if ( !settings_.collapseNearNotFlippable && collapsingFlippable && settings_.notFlippable && settings_.notFlippable->test( e ) )
            return { .status =  CollapseStatus::Flippable }; // cannot collapse a flippable edge incident to a not-flippable edge

        if ( !checkAspectRatios )
            continue;
        const auto pDest2 = mesh_.destPnt( topology.next( e ) );
        if ( eDest != vr )
        {
            const auto triAspect = triangleAspectRatio( collapsePos, pDest, pDest2 );
            maxNewAspectRatio = std::max( maxNewAspectRatio, triAspect );
            if ( settings_.preventNormalFlips )
            {
                auto da = cross( pDest - collapsePos, pDest2 - collapsePos );
                sumDblArea += Vector3d{ da };
                if ( triAspect < settings_.criticalTriAspectRatio ) // can trust direction of not-degenerate triangles only
                {
                    const auto oldA = cross( pDest - po, pDest2 - po );
                    if ( dot( da, oldA ) <= 0 )
                        triDblAreas_.push_back( da );
                }
            }
        }
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( po, pDest, pDest2 ) );
    }
    if ( oBdEdge
        && !smallShift( LineSegm3f{ po, mesh_.destPnt( oBdEdge ) }, collapsePos )
        && !smallShift( LineSegm3f{ po, mesh_.orgPnt( topology.prevLeftBd( oBdEdge ) ) }, collapsePos ) )
            return { .status =  CollapseStatus::PosFarBd }; // new vertex is too far from both existed boundary edges
    std::sort( originNeis_.begin(), originNeis_.end() );

    EdgeId dBdEdge; // a boundary edge !right(e) incident to dest( edgeToCollapse )
    for ( EdgeId e : orgRing0( topology, edgeToCollapse.sym() ) )
    {
        const auto eDest = topology.dest( e );
        assert ( eDest != vo );
        if ( std::binary_search( originNeis_.begin(), originNeis_.end(), eDest ) )
            return { .status =  CollapseStatus::MultipleEdge }; // to prevent appearance of multiple edges

        const auto pDest = mesh_.points[eDest];
        maxOldEdgeLenSq = std::max( maxOldEdgeLenSq, ( pd - pDest ).lengthSq() );
        maxNewEdgeLenSq = std::max( maxNewEdgeLenSq, ( collapsePos - pDest ).lengthSq() );
        if ( !topology.left( e ) )
        {
            dBdEdge = topology.next( e );
            assert( topology.isLeftBdEdge( dBdEdge ) );
            continue;
        }
        if ( !settings_.collapseNearNotFlippable && collapsingFlippable && settings_.notFlippable && settings_.notFlippable->test( e ) )
            return { .status =  CollapseStatus::Flippable }; // cannot collapse a flippable edge incident to a not-flippable edge

        if ( !checkAspectRatios )
            continue;
        const auto pDest2 = mesh_.destPnt( topology.next( e ) );
        if ( eDest != vl )
        {
            const auto triAspect = triangleAspectRatio( collapsePos, pDest, pDest2 );
            maxNewAspectRatio = std::max( maxNewAspectRatio, triAspect );
            if ( settings_.preventNormalFlips )
            {
                auto da = cross( pDest - collapsePos, pDest2 - collapsePos );
                sumDblArea += Vector3d{ da };
                if ( triAspect < settings_.criticalTriAspectRatio ) // can trust direction of not-degenerate triangles only
                {
                    const auto oldA = cross( pDest - pd, pDest2 - pd );
                    if ( dot( da, oldA ) <= 0 )
                        triDblAreas_.push_back( da );
                }
            }
        }
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( pd, pDest, pDest2 ) );
    }
    if ( dBdEdge
        && !smallShift( LineSegm3f{ pd, mesh_.destPnt( dBdEdge ) }, collapsePos )
        && !smallShift( LineSegm3f{ pd, mesh_.orgPnt( topology.prevLeftBd( dBdEdge ) ) }, collapsePos ) )
            return { .status =  CollapseStatus::PosFarBd }; // new vertex is too far from both existed boundary edges

    if ( vl && vr && oBdEdge && dBdEdge )
        return { .status =  CollapseStatus::TouchBd }; // prohibit collapse of an inner edge if it brings two boundaries in touch

    // special treatment for short edges on the last stage when collapse position is equal to one of edge's ends
    const bool tinyEdge = ( settings_.tinyEdgeLength >= 0 && ( po == collapsePos || pd == collapsePos ) )
        ? edgeLenSq <= sqr( settings_.tinyEdgeLength ) : false;
    if ( !tinyEdge && maxNewAspectRatio > maxOldAspectRatio && maxOldAspectRatio <= settings_.criticalTriAspectRatio )
        return { .status =  CollapseStatus::TriAspect }; // new triangle aspect ratio would be larger than all of old triangle aspect ratios and larger than allowed in settings

    if ( maxNewEdgeLenSq > maxOldEdgeLenSq )
        return { .status =  CollapseStatus::LongEdge }; // new edge would be longer than all of old edges and longer than allowed in settings

    // if at least one remaining triangle flips its normal, checks that new normal is consistent with the average normal of new vertex neighborhood
    if ( !triDblAreas_.empty() && ( ( po != pd ) || ( po != collapsePos ) ) )
    {
        auto n = Vector3f{ sumDblArea.normalized() };
        for ( const auto da : triDblAreas_ )
            if ( dot( da, n ) < 0 )
                return { .status =  CollapseStatus::NormalFlip };
    }

    if ( settings_.preCollapse && !settings_.preCollapse( edgeToCollapse, collapsePos ) )
        return { .status =  CollapseStatus::User }; // user prohibits the collapse

    assert( topology.org( edgeToCollapse ) == vo );
    return { .e = edgeToCollapse };
}

VertId MeshDecimator::forceCollapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos, const QuadraticForm3f & collapseForm )
{
    ++res_.vertsDeleted;

    auto & topology = mesh_.topology;
    const auto l = topology.left( edgeToCollapse );
    const auto r = topology.left( edgeToCollapse.sym() );
    if ( l )
        ++res_.facesDeleted;
    if ( r )
        ++res_.facesDeleted;

    const auto vo = topology.org( edgeToCollapse );
    mesh_.points[vo] = collapsePos;
    if ( settings_.region )
    {
        if ( l )
            settings_.region->reset( l );
        if ( r )
            settings_.region->reset( r );
    }
    auto eo = topology.collapseEdge( edgeToCollapse, onEdgeDel_ );
    if ( !eo )
        return {};

    if ( vo )
    {
        // must be done before computeQueueElement_ in addInQueueIfMissing_
        (*pVertForms_)[vo] = collapseForm;

        // update edges around remaining vertex
        for ( EdgeId e : orgRing( mesh_.topology, vo ) )
        {
            if ( addInQueueIfMissing_( e.undirected() ) && !outdated_.test_set( e.undirected() ) )
            {
                // collapse error for all the edges (including this one) incident to remaining vertex (vo) must increase,
                // we do not put the edge in the queue again before previous (smaller) error-version is extracted from the queue
                ++numOutdated_;
            }
            if ( mesh_.topology.left( e ) )
            {
                // an edge can be missed in the queue before because some criterion did not allow collapsing it,
                // and after just happened collapse that edge can become eligible again, so we try adding it in the queue;
                // but if the edge was already in the queue, assume that its error does not change so numOutdated_ is not modified
                addInQueueIfMissing_( mesh_.topology.prev( e.sym() ).undirected() );
            }
        }
    }
    return vo;
}

auto MeshDecimator::collapse_( EdgeId edgeToCollapse, const Vector3f & collapsePos, const QuadraticForm3f & collapseForm ) -> CollapseRes
{
    CanCollapseRes can = canCollapse_( edgeToCollapse, collapsePos );
    if ( can.status != CollapseStatus::Ok )
        return { .status = can.status };
    assert( can.e == edgeToCollapse || can.e == edgeToCollapse.sym() );
    return { .v = forceCollapse_( can.e, collapsePos, collapseForm ) };
}

static void packMesh( Mesh & mesh, const DecimateSettings & settings,
    FaceMap * outFmap = nullptr, VertMap * outVmap = nullptr, WholeEdgeMap * outEmap = nullptr )
{
    MR_TIMER;
    FaceMap fmap;
    FaceMap *pFmap = settings.region ? &fmap : outFmap;
    VertMap vmap;
    VertMap *pVmap = settings.vertForms ? &vmap : outVmap;
    WholeEdgeMap emap;
    WholeEdgeMap *pEmap = settings.notFlippable ? &emap : outEmap;
    mesh.pack( pFmap, pVmap, pEmap );

    if ( settings.region )
        *settings.region = settings.region->getMapping( fmap, mesh.topology.faceSize() );

    if ( settings.vertForms )
    {
        auto & vertForms = *settings.vertForms;
        for ( VertId oldV{ 0 }; oldV < vmap.size(); ++oldV )
            if ( auto newV = vmap[oldV] )
                if ( newV < oldV )
                    vertForms[newV] = vertForms[oldV];
        vertForms.resize( mesh.topology.vertSize() );
    }

    auto packedUndirectedEdgeId = [&emap]( UndirectedEdgeId unpackedId )
    {
        auto packedEdgeId = emap[unpackedId];
        return packedEdgeId ? packedEdgeId.undirected() : UndirectedEdgeId();
    };

    if ( settings.notFlippable )
        *settings.notFlippable = settings.notFlippable->getMapping( packedUndirectedEdgeId, mesh.topology.undirectedEdgeSize() );

    if ( settings.edgesToCollapse )
        *settings.edgesToCollapse = settings.edgesToCollapse->getMapping( packedUndirectedEdgeId, mesh.topology.undirectedEdgeSize() );

    if ( settings.twinMap )
    {
        UndirectedEdgeHashMap packedTwinMap;
        packedTwinMap.reserve( settings.twinMap->size() );

        for ( const auto& [key, value] : *settings.twinMap )
        {
            auto packedKey = packedUndirectedEdgeId( key );
            auto packedValue = packedUndirectedEdgeId( value );
            if ( packedKey && packedValue )
                packedTwinMap[packedKey] = packedValue;
            else
                assert( !packedKey && !packedValue );
        }

        *settings.twinMap = std::move( packedTwinMap );
    }

    if ( settings.bdVerts )
        *settings.bdVerts = settings.bdVerts->getMapping( vmap, mesh.topology.vertSize() );

    if ( outFmap && outFmap != pFmap )
        *outFmap = std::move( *pFmap );
    if ( outVmap && outVmap != pVmap )
        *outVmap = std::move( *pVmap );
    if ( outEmap && outEmap != pEmap )
        *outEmap = std::move( *pEmap );
}

DecimateResult MeshDecimator::run()
{
    MR_TIMER;

    if ( settings_.bdVerts )
        pBdVerts_ = settings_.bdVerts;
    else
    {
        pBdVerts_ = &myBdVerts_;
        if ( !settings_.touchNearBdEdges )
            myBdVerts_ = getBoundaryVerts( mesh_.topology, settings_.region );
    }

    if ( !initialize_() )
        return res_;

    res_.errorIntroduced = settings_.maxError;
    int lastProgressFacesDeleted = 0;
    const int maxFacesDeleted = std::min(
        settings_.region ? (int)settings_.region->count() : mesh_.topology.numValidFaces(), settings_.maxDeletedFaces );
    while ( !queue_.empty() )
    {
        // update queue elements if there is significant portion of outdated edges there
        if ( queue_.size() >= 1024 && queue_.size() < 5 * numOutdated_ )
        {
            updateQueue_();
            if ( queue_.empty() )
                break; // if old queue was filled only with invalid elements
        }
        const auto topQE = queue_.top();
        auto ue = topQE.uedgeId();
        queue_.pop();
        if ( res_.facesDeleted >= settings_.maxDeletedFaces || res_.vertsDeleted >= settings_.maxDeletedVertices )
        {
            res_.errorIntroduced = std::sqrt( topQE.c );
            break;
        }

        if ( settings_.progressCallback && res_.facesDeleted >= 1000 + lastProgressFacesDeleted )
        {
            if ( !settings_.progressCallback( 0.25f + 0.75f * res_.facesDeleted / maxFacesDeleted ) )
                return res_;
            lastProgressFacesDeleted = res_.facesDeleted;
        }

        if ( !validInQueue_.test( ue ) )
        {
            // edge has been deleted by this moment
            assert( mesh_.topology.isLoneEdge( ue ) );
            continue;
        }
        assert( !mesh_.topology.isLoneEdge( ue ) );

        QuadraticForm3f collapseForm;
        Vector3f collapsePos;
        auto qe = computeQueueElement_( ue, topQE.x.edgeOp == EdgeOp::CollapseOptPos, &collapseForm, &collapsePos );
        if ( !qe )
        {
            validInQueue_.reset( ue );
            continue;
        }

        if ( qe->c > topQE.c )
        {
            queue_.push( *qe );
            if ( outdated_.test_set( ue, false ) )
                --numOutdated_;
            continue;
        }

        validInQueue_.reset( ue );

        if ( qe->x.edgeOp == EdgeOp::Flip )
        {
            flipEdge_( ue );
            if ( settings_.twinMap )
                if ( auto twin = getAt( *settings_.twinMap, ue ) )
                    flipEdge_( twin );
        }
        else
        {
            // edge collapse
            auto canCollapseRes = canCollapse_( ue, collapsePos );
            if ( canCollapseRes.status != CollapseStatus::Ok )
            {
                if ( topQE.x.edgeOp == EdgeOp::CollapseOptPos && geomFail_( canCollapseRes.status ) )
                {
                    addInQueue_( ue, false );
                    continue;
                }
                if ( canCollapseRes.status != CollapseStatus::MultipleEdge )
                    continue;

                const bool tinyEdge = ( settings_.tinyEdgeLength >= 0 && topQE.x.edgeOp == EdgeOp::CollapseEnd )
                    ? mesh_.edgeLengthSq( ue ) <= sqr( settings_.tinyEdgeLength ) : false;
                if ( !tinyEdge )
                    continue;

                // if the collapse of a tiny edge failed because of appearance of multiple edges,
                // test whether left or right triangle contains single 3-degree vertex inside, then collapse it instead
                const auto en = mesh_.topology.next( ue );
                const auto ep = mesh_.topology.prev( ue );
                // the check that left/right triangle exists is done by isOrgInnerAndHasDegree
                ue = {};
                if ( !ue && mesh_.topology.isOrgInnerAndHasDegree( en.sym(), 3 ) )
                {
                    auto vo = mesh_.topology.org( en );
                    collapsePos = mesh_.points[vo];
                    collapseForm = (*pVertForms_)[vo];
                    canCollapseRes = CanCollapseRes{ en };
                    ue = en;
                }
                if ( !ue && mesh_.topology.isOrgInnerAndHasDegree( ep.sym(), 3 ) )
                {
                    auto vo = mesh_.topology.org( ep );
                    collapsePos = mesh_.points[vo];
                    collapseForm = (*pVertForms_)[vo];
                    canCollapseRes = CanCollapseRes{ ep };
                    ue = ep;
                }
                if ( !ue )
                    continue;
            }
            if ( settings_.twinMap )
                if ( auto twin = getAt( *settings_.twinMap, ue ) )
                {
                    const auto twinCollapseForm = collapseForm_( twin, collapsePos );
                    const auto twinCollapseRes = collapse_( twin, collapsePos, twinCollapseForm );
                    if ( twinCollapseRes.status != CollapseStatus::Ok )
                        continue;
                }
            forceCollapse_( canCollapseRes.e, collapsePos, collapseForm );
        }
    }

    if ( settings_.progressCallback && !settings_.progressCallback( 1.0f ) )
        return res_;

    if ( settings_.packMesh )
        packMesh( mesh_, settings_ );
    res_.cancelled = false;
    return res_;
}

static DecimateResult decimateMeshSerial( Mesh & mesh, const DecimateSettings & settings, const UndirectedEdgeBitSet * initialEdges = nullptr )
{
    MR_TIMER;
    if ( settings.maxDeletedFaces <= 0 || settings.maxDeletedVertices <= 0 )
    {
        DecimateResult res;
        res.cancelled = false;
        return res;
    }
    MeshDecimator md( mesh, settings, initialEdges );
    return md.run();
}

/// all edges incident to given vertices
static UndirectedEdgeBitSet getIncidentEdges( const MeshTopology & topology, const VertBitSet & verts )
{
    UndirectedEdgeBitSet res( topology.undirectedEdgeSize() );
    for ( auto v : verts )
        for ( auto e : orgRing( topology, v ) )
            res.set( e.undirected() );
    return res;
}

FaceBitSet getSubdividePart( const FaceBitSet & valids, size_t subdivideParts, size_t myPart )
{
    assert( subdivideParts > 1 );
    const auto sz = subdivideParts;
    assert( myPart < sz );
    const auto facesPerPart = ( valids.size() / ( sz * FaceBitSet::bits_per_block ) ) * FaceBitSet::bits_per_block;

    const auto fromFace = myPart * facesPerPart;
    const auto toFace = ( myPart + 1 ) < sz ? ( myPart + 1 ) * facesPerPart : valids.size();
    FaceBitSet res( toFace );
    res.set( FaceId{ fromFace }, toFace - fromFace, true );
    res &= valids;
    return res;
}

/// returns the faces of the subtrees of mesh's AABB tree (or of a tree constructed for the region), at least subdivideParts of them
static Vector<FaceBitSet> getTreeParts( const Mesh & mesh, const DecimateSettings & settings )
{
    MR_TIMER;
    std::optional<AABBTree> ownTree;
    const AABBTree * tree = mesh.getAABBTreeNotCreate();
    if ( !tree )
        tree = &ownTree.emplace( MeshPart{ mesh, settings.region } );
    const auto roots = tree->getSubtrees( settings.subdivideParts );
    Vector<FaceBitSet> res( roots.size() );
    ParallelFor( res, [&]( size_t i )
    {
        res[i] = tree->getSubtreeLeaves( roots[i] );
        res[i].resize( mesh.topology.faceSize() );
    } );
    return res;
}

/// \param treeParts if not null, the faces of each part, used instead of getSubdividePart
static DecimateResult decimateMeshParallelInplace( MR::Mesh & mesh, const DecimateSettings & settings, Vector<FaceBitSet> * treeParts = nullptr )
{
    MR_TIMER;
    assert( settings.subdivideParts > 1 );
    assert( !treeParts || ( !settings.partFaces && treeParts->size() > 1 ) );
    const int sz = treeParts ? int( treeParts->size() ) : settings.subdivideParts;
    // the vertices of the region, which are distributed among the parts to determine their quotas of deletions
    const auto numIniVerts = settings.region ? getIncidentVerts( mesh.topology, *settings.region ).count() : mesh.topology.numValidVerts();
    assert( !settings.partFaces || settings.partFaces->size() == sz );

    DecimateResult res;
    if ( mesh.topology.getFaceIds( settings.region ).none() )
    {
        // nothing to decimate
        res.cancelled = false;
        return res;
    }

    if ( settings.progressCallback && !settings.progressCallback( 0 ) )
        return res;

    // if decimateBetweenParts, then parts are the leaves of a binary tree, and as soon as two sibling nodes are decimated,
    // their union is decimated in parallel with other nodes; only the root (the whole mesh) is decimated at the end alone
    const bool hierarchical = settings.decimateBetweenParts;

    /// a subdivision part (first sz elements) or an internal node of the tree in hierarchical mode
    struct alignas(64) Parts
    {
        /// region faces of the subdivision part
        FaceBitSet region;

        /// vertices to be fixed during subdivision of individual parts
        VertBitSet bdVerts;

        /// these are inner vertices of this part that can be easily deleted during part decimation;
        /// filled only if limitedDeletion, and only for subdivision parts
        VertBitSet removableVerts;

        /// edges incident to region faces
        UndirectedEdgeBitSet regionEdges;

        DecimateResult decimRes;

        /// this node covers subdivision parts [beginPart, endPart)
        int beginPart = 0, endPart = 0;

        /// the links in the tree, -1 if absent
        int parent = -1, left = -1, right = -1;

        /// the number of removable vertices in this node, which become removable only after the union of its children
        std::atomic<size_t> extraRemovableVerts{ 0 };

        /// the number of removable vertices in all parts of this node, and the number of deleted vertices/faces in all nodes of this subtree
        size_t numRemovableVerts = 0;
        int subtreeVertsDeleted = 0, subtreeFacesDeleted = 0;

        /// the node is decimated by the thread which finished the second of its children
        std::atomic<int> finishedChildren{ 0 };
    };
    Vector<Parts> parts( hierarchical ? 2 * sz - 1 : sz );
    for ( int i = 0; i < sz; ++i )
    {
        parts[i].beginPart = i;
        parts[i].endPart = i + 1;
    }

    // build the tree with internal nodes after the leaves, a child always has smaller id than its parent
    int root = -1;
    if ( hierarchical )
    {
        int nextNode = sz;
        auto build = [&]( auto && self, int beginPart, int endPart ) -> int
        {
            if ( endPart - beginPart == 1 )
                return beginPart;
            const int midPart = beginPart + ( endPart - beginPart ) / 2;
            const int left = self( self, beginPart, midPart );
            const int right = self( self, midPart, endPart );
            const int id = nextNode++;
            auto & node = parts[id];
            node.beginPart = beginPart;
            node.endPart = endPart;
            node.left = left;
            node.right = right;
            parts[left].parent = parts[right].parent = id;
            return id;
        };
        root = build( build, 0, sz );
        assert( root + 1 == parts.size() );
    }

    // returns the smallest node containing all parts in [loPart, hiPart]
    auto smallestNode = [&]( int loPart, int hiPart )
    {
        assert( 0 <= loPart && loPart <= hiPart && hiPart < sz );
        int id = root;
        for ( ;; )
        {
            const auto & node = parts[id];
            if ( node.left < 0 )
                return id;
            if ( hiPart < parts[node.left].endPart )
                id = node.left;
            else if ( loPart >= parts[node.right].beginPart )
                id = node.right;
            else
                return id;
        }
    };

    // determine faces for each part
    ParallelFor( 0, sz, [&]( size_t i )
    {
        if ( settings.partFaces )
            parts[i].region = std::move( (*settings.partFaces)[i] );
        else if ( treeParts )
            parts[i].region = std::move( (*treeParts)[i] );
        else
            parts[i].region = getSubdividePart( mesh.topology.getValidFaces(), settings.subdivideParts, i );
    } );
    if ( settings.progressCallback && !settings.progressCallback( 0.03f ) )
        return res;

    // the part of each face (parts do not overlap), -1 for faces outside all parts
    Vector<int, FaceId> facePart( mesh.topology.faceSize(), -1 );
    ParallelFor( 0, sz, [&]( size_t i )
    {
        for ( auto f : parts[i].region )
            facePart[f] = int( i );
    } );

    // determine edges in between the parts
    UndirectedEdgeBitSet stableEdges( mesh.topology.undirectedEdgeSize() );
    BitSetParallelForAll( stableEdges, [&]( UndirectedEdgeId ue )
    {
        FaceId l = mesh.topology.left( ue );
        FaceId r = mesh.topology.right( ue );
        if ( l && r && facePart[l] != facePart[r] )
            stableEdges.set( ue );
    } );
    if ( settings.progressCallback && !settings.progressCallback( 0.07f ) )
        return res;

    const bool limitedDeletion =
        settings.maxDeletedVertices < mesh.topology.numValidVerts() ||
        settings.maxDeletedFaces < mesh.topology.numValidFaces();

    ParallelFor( 0, sz, [&]( size_t i )
    {
        if ( settings.region )
            parts[i].region &= *settings.region;
        parts[i].bdVerts.resize( mesh.topology.vertSize() );
        if ( limitedDeletion )
            parts[i].removableVerts.resize( mesh.topology.vertSize() );
    } );

    // the range of parts of the faces around a vertex or a face:
    // lo = -1 if it has a face outside of all parts (or a hole or a face outside of region if !touchNearBdEdges), so it is fixed till the final decimation,
    // hi = INT_MAX if it has no region faces around, so it cannot be deleted
    struct PartRange
    {
        int lo = INT_MAX, hi = -1;
        void include( const PartRange & r ) { lo = std::min( lo, r.lo ); hi = std::max( hi, r.hi ); }
        bool inside( const Parts & node ) const { return node.beginPart <= lo && hi < node.endPart; }
    };
    Vector<PartRange, VertId> vertRange( mesh.topology.vertSize() );
    // vertices on the boundaries of parts
    tbb::enumerable_thread_specific<Vector<VertId>> threadBdVerts;

    // bdVerts: vertices with incident (region) faces from several parts, and if !touchNearBdEdges also on hole or region boundaries;
    // one pass for all parts instead of a whole-mesh pass per part; writing different VertBitSets by v is race-free
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        PartRange r;
        bool outside = false, inRegion = false;
        for ( auto e : orgRing( mesh.topology, v ) )
        {
            const auto f = mesh.topology.left( e );
            const bool fInRegion = f && ( !settings.region || settings.region->test( f ) );
            if ( fInRegion )
                inRegion = true;
            else if ( !settings.touchNearBdEdges )
                outside = true;
            if ( !f )
                continue;
            if ( const int p = facePart[f]; p >= 0 )
                r.include( { p, p } );
            else
                outside = true;
        }
        if ( outside )
            r.lo = -1;
        if ( !inRegion )
            r.hi = INT_MAX;
        vertRange[v] = r;
        if ( r.lo == r.hi )
            return; // inside one part, or no faces at all
        for ( auto e : orgRing( mesh.topology, v ) )
            if ( const auto f = mesh.topology.left( e ); f && facePart[f] >= 0 && ( settings.touchNearBdEdges || contains( settings.region, f ) ) )
                parts[facePart[f]].bdVerts.set( v );
        if ( hierarchical )
            threadBdVerts.local().push_back( v );
    } );

    if ( limitedDeletion )
    {
        // removableVerts: vertices of parts excluding the first level of inner vertices, because they cannot be all deleted without violating aspect ratio criterion;
        // extraRemovableVerts: such vertices in a node, but not in its children
        Vector<PartRange, FaceId> faceRange( mesh.topology.faceSize() );
        BitSetParallelFor( mesh.topology.getValidFaces(), [&]( FaceId f )
        {
            PartRange r;
            for ( auto e : leftRing( mesh.topology, f ) )
                r.include( vertRange[mesh.topology.org( e )] );
            faceRange[f] = r;
        } );
        BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
        {
            PartRange r;
            for ( auto e : orgRing( mesh.topology, v ) )
            {
                const auto f = mesh.topology.left( e );
                if ( !f )
                    return;
                r.include( faceRange[f] );
            }
            if ( r.lo < 0 || r.lo > r.hi || r.hi >= sz )
                return;
            if ( r.lo == r.hi )
                parts[r.lo].removableVerts.set( v );
            else if ( hierarchical )
                parts[smallestNode( r.lo, r.hi )].extraRemovableVerts.fetch_add( 1, std::memory_order_relaxed );
        } );
    }

    // edges of each part's region (facePart limited to settings.region) without edges touching part's bdVerts,
    // in one pass over all edges instead of a getIncidentEdges pass per part;
    // all edge bit sets have the same indexing, so threads write disjoint blocks
    ParallelFor( 0, sz, [&]( size_t i ) { parts[i].regionEdges.resize( mesh.topology.undirectedEdgeSize() ); } );
    BitSetParallelForAll( parts[0].regionEdges, [&]( UndirectedEdgeId ue )
    {
        for ( auto f : { mesh.topology.left( ue ), mesh.topology.right( ue ) } )
        {
            if ( !f || facePart[f] < 0 || ( settings.region && !settings.region->test( f ) ) )
                continue;
            auto & part = parts[size_t( facePart[f] )];
            if ( !part.bdVerts.test( mesh.topology.org( ue ) ) && !part.bdVerts.test( mesh.topology.dest( ue ) ) )
                part.regionEdges.set( ue );
        }
    } );

    if ( settings.progressCallback && !settings.progressCallback( 0.14f ) )
        return res;

    mesh.topology.preferEdges( stableEdges );
    if ( hierarchical )
    {
        // the edge of a vertex can be deleted only in the node, where the vertex is not fixed anymore;
        // otherwise the edge of a vertex could change during decimation of one node, while another node reads it
        Vector<int> depth( parts.size(), 0 );
        for ( int id = root - 1; id >= 0; --id )
            depth[id] = depth[parts[id].parent] + 1;
        // the depth of the node, where the edge or the vertex stops being fixed
        auto edgeDepth = [&]( EdgeId e )
        {
            const auto l = mesh.topology.left( e );
            const auto r = mesh.topology.right( e );
            const int pl = facePart[l ? l : r]; // a boundary edge belongs to one part
            const int pr = facePart[r ? r : l];
            if ( pl < 0 || pr < 0 )
                return 0;
            return depth[smallestNode( std::min( pl, pr ), std::max( pl, pr ) )];
        };
        auto vertDepth = [&]( VertId v )
        {
            const auto r = vertRange[v];
            return ( r.lo < 0 || r.hi >= sz ) ? 0 : depth[smallestNode( r.lo, r.hi )];
        };
        Vector<VertId> bdVerts;
        for ( auto & tv : threadBdVerts )
            bdVerts.insert( bdVerts.end(), tv.begin(), tv.end() );
        Vector<EdgeId> betterEdges( bdVerts.size() );
        ParallelFor( bdVerts, [&]( size_t i )
        {
            const auto v = bdVerts[i];
            const int vd = vertDepth( v );
            int bestDepth = edgeDepth( mesh.topology.edgeWithOrg( v ) );
            for ( auto e : orgRing( mesh.topology, v ) )
            {
                if ( bestDepth <= vd )
                    break;
                if ( const int d = edgeDepth( e ); d < bestDepth )
                {
                    bestDepth = d;
                    betterEdges[i] = e;
                }
            }
        } );
        for ( auto e : betterEdges )
        {
            if ( !e )
                continue;
            // make e the edge of its origin
            const auto v = mesh.topology.org( e );
            mesh.topology.setOrg( e, {} );
            mesh.topology.setOrg( e, v );
        }
    }
    if ( settings.progressCallback && !settings.progressCallback( 0.16f ) )
        return res;

    // compute quadratic form in each vertex
    Vector<QuadraticForm3f, VertId> mVertForms;
    if ( settings.vertForms )
        mVertForms = std::move( *settings.vertForms );
    if ( mVertForms.empty() )
        mVertForms = computeFormsAtVertices( MeshPart{ mesh, settings.region }, settings.stabilizer, settings.angleWeightedDistToPlane, settings.notFlippable );
    if ( settings.progressCallback && !settings.progressCallback( 0.2f ) )
        return res;

    mesh.topology.stopUpdatingValids();
    TbbThreadMutex reporterMutex;
    std::atomic<bool> cancelled{ false };
    std::atomic<int> finishedParts{ 0 };
    const int numToDecimate = hierarchical ? root : sz; // all nodes except the root
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, sz ), [&]( const tbb::blocked_range<size_t>& range )
    {
        const auto reporterLock = reporterMutex.tryLock();
        const bool reportProgressFromThisThread = settings.progressCallback && reporterLock;
        auto reportThreadProgress = [&]( float p )
        {
            if ( cancelled.load( std::memory_order_relaxed ) )
                return false;
            if ( reportProgressFromThisThread && !settings.progressCallback( 0.2f + 0.65f * ( finishedParts.load( std::memory_order_relaxed ) + p ) / numToDecimate ) )
            {
                cancelled.store( true, std::memory_order_relaxed );
                return false;
            }
            return true;
        };

        // decimates given part or internal node, whose children are already decimated; returns false if cancelled
        auto decimateNode = [&]( int id )
        {
            if ( !reportThreadProgress( 0 ) )
                return false;
            auto & node = parts[id];
            UndirectedEdgeBitSet seamEdges;
            if ( node.left >= 0 )
            {
                // unite the children, their common boundary is not fixed anymore
                auto & l = parts[node.left];
                auto & r = parts[node.right];
                node.region = std::exchange( l.region, {} );
                node.region |= r.region;
                r.region = {};
                // region edges of the children (may include the ones deleted there) without edges near children's bdVerts
                node.regionEdges = std::exchange( l.regionEdges, {} );
                node.regionEdges |= r.regionEdges;
                r.regionEdges = {};
                node.bdVerts = std::exchange( l.bdVerts, {} );
                node.bdVerts |= r.bdVerts;
                r.bdVerts = {};
                Vector<VertId> freedVerts;
                for ( auto v : node.bdVerts )
                    if ( vertRange[v].inside( node ) )
                        freedVerts.push_back( v );
                for ( auto v : freedVerts )
                    node.bdVerts.reset( v );
                // only the edges near freed vertices were not considered in the children
                seamEdges.resize( mesh.topology.undirectedEdgeSize() );
                for ( auto v : freedVerts )
                {
                    for ( auto e : orgRing( mesh.topology, v ) )
                    {
                        seamEdges.set( e.undirected() );
                        if ( !node.bdVerts.test( mesh.topology.dest( e ) )
                            && ( contains( node.region, mesh.topology.left( e ) ) || contains( node.region, mesh.topology.right( e ) ) ) )
                            node.regionEdges.set( e.undirected() );
                    }
                }
                node.numRemovableVerts = l.numRemovableVerts + r.numRemovableVerts + node.extraRemovableVerts.load( std::memory_order_relaxed );
                node.subtreeVertsDeleted = l.subtreeVertsDeleted + r.subtreeVertsDeleted;
                node.subtreeFacesDeleted = l.subtreeFacesDeleted + r.subtreeFacesDeleted;
            }
            else if ( limitedDeletion )
                node.numRemovableVerts = node.removableVerts.count();

            DecimateSettings subSeqSettings = settings;
            if ( limitedDeletion )
            {
                /// give quota of deletions proportional to the number of removable vertices in the node, minus already deleted in its children
                const auto partFraction = float( node.numRemovableVerts ) / numIniVerts;
                subSeqSettings.maxDeletedVertices = std::max( 0, int( settings.maxDeletedVertices * partFraction ) - node.subtreeVertsDeleted );
                subSeqSettings.maxDeletedFaces = std::max( 0, int( settings.maxDeletedFaces * partFraction ) - node.subtreeFacesDeleted );
            }
            if ( settings.minFacesInPart > 0 )
            {
                const int startFaces = (int)node.region.count();
                const int minFaces = settings.minFacesInPart * ( node.endPart - node.beginPart );
                if ( startFaces > minFaces )
                    subSeqSettings.maxDeletedFaces = std::min( subSeqSettings.maxDeletedFaces, startFaces - minFaces );
                else
                    subSeqSettings.maxDeletedFaces = 0;
            }
            subSeqSettings.packMesh = false;
            subSeqSettings.vertForms = &mVertForms;
            // regionEdges already exclude the edges touching bdVerts
            subSeqSettings.touchNearBdEdges = true;
            // after mesh.topology.stopUpdatingValids(), seq-subdivider cannot find bdVerts by itself
            subSeqSettings.bdVerts = &node.bdVerts;
            subSeqSettings.regionEdges = &node.regionEdges;
            subSeqSettings.region = &node.region;
            if ( reportProgressFromThisThread )
                subSeqSettings.progressCallback = [reportThreadProgress]( float p ) { return reportThreadProgress( p ); };
            else if ( settings.progressCallback )
                subSeqSettings.progressCallback = [&cancelled]( float ) { return !cancelled.load( std::memory_order_relaxed ); };
            // a part puts in its initial queue all its region edges, found by visiting only the set bits (and not by testing every edge of the mesh);
            // a united node only the edges near the former border between its children
            node.decimRes = decimateMeshSerial( mesh, subSeqSettings, node.left >= 0 ? &seamEdges : &node.regionEdges );
            node.subtreeVertsDeleted += node.decimRes.vertsDeleted;
            node.subtreeFacesDeleted += node.decimRes.facesDeleted;

            if ( node.decimRes.cancelled || !reportThreadProgress( 1 ) )
                return false;
            finishedParts.fetch_add( 1, std::memory_order_relaxed );
            return true;
        };

        for ( size_t i = range.begin(); i < range.end(); ++i )
        {
            if ( !decimateNode( int( i ) ) )
                break;
            // go up the tree while this thread finishes the second child of a node
            bool ok = true;
            for ( int id = parts[i].parent; ok && id >= 0 && id != root && parts[id].finishedChildren.fetch_add( 1 ) == 1; id = parts[id].parent )
                ok = decimateNode( id );
            if ( !ok )
                break;
        }
    } );

    if ( settings.region )
    {
        // not ParallelFor to allow for subdivisions not aligned to BitSet's block boundaries;
        // united regions of decimated nodes are stored in the nodes, and the regions of their children are empty
        *settings.region = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, parts.size() ), FaceBitSet{},
            [&] ( const tbb::blocked_range<size_t> range, FaceBitSet cur )
            {
                for ( size_t i = range.begin(); i < range.end(); i++ )
                    cur |= parts[i].region;
                return cur;
            },
            [&] ( FaceBitSet a, const FaceBitSet& b )
            {
                a |= b;
                return a;
            } );
    }

    // restore valids computation before return even if the operation was canceled
    mesh.topology.computeValidsFromEdges();

    if ( cancelled.load( std::memory_order_relaxed ) || ( settings.progressCallback && !settings.progressCallback( 0.9f ) ) )
        return res;

    if ( settings.partFaces )
    {
        assert( !settings.packMesh ); // otherwise, returned partFaces will contain wrong distribution of faces
        assert( !settings.decimateBetweenParts ); // otherwise, returned partFaces will contain some deleted faces, and some faces can actually be in-between original parts
        for ( int i = 0; i < sz; ++i )
            (*settings.partFaces)[i] = std::move( parts[i].region );
    }

    DecimateSettings seqSettings = settings;
    for ( const auto & submesh : parts )
    {
        seqSettings.maxDeletedFaces -= submesh.decimRes.facesDeleted;
        seqSettings.maxDeletedVertices -= submesh.decimRes.vertsDeleted;
    }
    seqSettings.vertForms = &mVertForms;
    seqSettings.progressCallback = subprogress( settings.progressCallback, 0.9f, 1.0f );
    if ( hierarchical )
    {
        // all edges except for the ones near the boundary between two halves of the mesh were already considered in the tree's nodes
        auto & rootNode = parts[root];
        auto seamVerts = std::exchange( parts[rootNode.left].bdVerts, {} );
        seamVerts |= parts[rootNode.right].bdVerts;
        const auto seamEdges = getIncidentEdges( mesh.topology, seamVerts );
        seqSettings.packMesh = false;
        seqSettings.progressCallback = subprogress( settings.progressCallback, 0.9f, 0.95f );
        res = decimateMeshSerial( mesh, seqSettings, &seamEdges );
        if ( !res.cancelled && limitedDeletion && res.facesDeleted < seqSettings.maxDeletedFaces && res.vertsDeleted < seqSettings.maxDeletedVertices )
        {
            // some nodes did not reach their quotas, so others shall delete more
            seqSettings.maxDeletedFaces -= res.facesDeleted;
            seqSettings.maxDeletedVertices -= res.vertsDeleted;
            seqSettings.packMesh = settings.packMesh;
            seqSettings.progressCallback = subprogress( settings.progressCallback, 0.95f, 1.0f );
            const auto seamRes = res;
            res = decimateMeshSerial( mesh, seqSettings );
            res.facesDeleted += seamRes.facesDeleted;
            res.vertsDeleted += seamRes.vertsDeleted;
        }
        else if ( !res.cancelled )
        {
            if ( settings.progressCallback && !settings.progressCallback( 1.0f ) )
                res.cancelled = true;
            else if ( settings.packMesh )
                packMesh( mesh, seqSettings );
        }
    }
    else if ( settings.decimateBetweenParts )
    {
        res = decimateMeshSerial( mesh, seqSettings );
    }
    else
    {
        if ( seqSettings.packMesh )
            packMesh( mesh, seqSettings );
        res.cancelled = false;
    }

    // update res from submesh decimations
    for ( const auto & submesh : parts )
    {
        res.facesDeleted += submesh.decimRes.facesDeleted;
        res.vertsDeleted += submesh.decimRes.vertsDeleted;
    }

    if ( settings.vertForms )
        *settings.vertForms = std::move( mVertForms );
    return res;
}

DecimateResult decimateMesh( Mesh & mesh, const DecimateSettings & settings0 )
{
    auto settings = settings0;
    if ( settings.maxError >= FLT_MAX &&
         settings.maxEdgeLen >= FLT_MAX &&
         settings.maxDeletedFaces >= INT_MAX &&
         settings.maxDeletedVertices >= INT_MAX )
    {
        assert( false ); // below logic can be deleted in future
        // if unexperienced user started decimation with default settings, then decimate half of faces
        settings.maxDeletedFaces = int( settings.region ? settings.region->count() : mesh.topology.numValidFaces() ) / 2;
    }

    DecimateResult res;
#ifndef NDEBUG
    if ( !mesh.topology.checkValidity() )
        return res;
#endif

    Vector<FaceBitSet> treeParts;
    if ( settings.subdivideParts > 1 && !settings.partFaces && settings.partsMode == DecimatePartsMode::AABBTree )
        treeParts = getTreeParts( mesh, settings ); // before invalidateCaches() to use the tree of the mesh if any

    mesh.invalidateCaches(); // free memory occupied by trees before running the algorithm, which makes them invalid anyway
    if ( settings.subdivideParts <= 1 || ( settings.partsMode == DecimatePartsMode::AABBTree && !settings.partFaces && treeParts.size() <= 1 ) )
        res = decimateMeshSerial( mesh, settings );
    else
        res = decimateMeshParallelInplace( mesh, settings, treeParts.empty() ? nullptr : &treeParts );
    assert ( !mesh.getAABBTreeNotCreate() ); // make sure that nobody created the tree by mistake
    assert ( mesh.topology.checkValidity() );
    return res;
}

DecimateResult decimateObjectMeshData( ObjectMeshData & data, const DecimateSettings & set0 )
{
    MR_TIMER;
    DecimateResult res;
    if ( !data.mesh )
    {
        assert( false );
        return res;
    }

    DecimateSettings settings = set0;

    const bool finalMeshPack = settings.packMesh;
    settings.packMesh = false;

    if ( settings.subdivideParts > 1 )
        settings.progressCallback = subprogress( set0.progressCallback, 0.2f, 1.0f );

    resizeAttributesToMesh( data );
    const bool updateUV = !data.uvCoordinates.empty();
    const bool updateColorMap = !data.vertColors.empty();
    const bool updateFaceColorMap = !data.faceColors.empty();
    const bool updateTexturePerFace = !data.texturePerFace.empty();

    if ( updateUV || updateColorMap )
    {
        MeshAttributesToUpdate meshParams;
        if ( updateUV )
            meshParams.uvCoords = &data.uvCoordinates;
        if ( updateColorMap )
            meshParams.colorMap = &data.vertColors;
        settings.preCollapse = meshPreCollapseVertAttribute( *data.mesh, meshParams );
    }

    std::shared_ptr<UndirectedEdgeBMap> emap;
    if ( settings.subdivideParts > 1 && settings.partsMode != DecimatePartsMode::AABBTree )
    {
        auto packMapping = data.mesh->packOptimally( false );
        if ( updateUV )
            data.uvCoordinates = rearrangeVectorByMap( data.uvCoordinates, packMapping.v );
        if ( updateColorMap )
            data.vertColors = rearrangeVectorByMap( data.vertColors, packMapping.v );
        if ( updateFaceColorMap )
            data.faceColors = rearrangeVectorByMap( data.faceColors, packMapping.f );
        if ( updateTexturePerFace )
            data.texturePerFace = rearrangeVectorByMap( data.texturePerFace, packMapping.f );
        if ( data.selectedFaces.any() )
            data.selectedFaces = data.selectedFaces.getMapping( packMapping.f );
        if ( settings.region )
            *settings.region = settings.region->getMapping( packMapping.f );
        emap = std::allocate_shared<UndirectedEdgeBMap>( Allocator<UndirectedEdgeBMap>(), std::move( packMapping.e ) );
        if ( !reportProgress( set0.progressCallback, .2f ) )
            return res;
    }

    res = decimateMesh( *data.mesh, settings );
    if ( res.cancelled )
        return res;

    if ( finalMeshPack )
    {
        auto packMapping = data.mesh->packOptimally( false );
        if ( updateUV )
            data.uvCoordinates = rearrangeVectorByMap( data.uvCoordinates, packMapping.v );
        if ( updateColorMap )
            data.vertColors = rearrangeVectorByMap( data.vertColors, packMapping.v );
        if ( updateFaceColorMap )
            data.faceColors = rearrangeVectorByMap( data.faceColors, packMapping.f );
        if ( updateTexturePerFace )
            data.texturePerFace = rearrangeVectorByMap( data.texturePerFace, packMapping.f );
        if ( data.selectedFaces.any() )
            data.selectedFaces = data.selectedFaces.getMapping( packMapping.f );
        if ( settings.region )
            *settings.region = settings.region->getMapping( packMapping.f );
        if ( emap )
            *emap = compose( packMapping.e, *emap );
        else
            emap = std::allocate_shared<UndirectedEdgeBMap>( Allocator<UndirectedEdgeBMap>(), std::move( packMapping.e ) );

        packMapping = {}; //free memory
        data.mesh->shrinkToFit();
    }
    else
    {
        data.selectedFaces &= data.mesh->topology.getValidFaces();
    }

    if ( emap && emap->tsize > 0 )
    {
        data.selectedEdges = mapEdges( *emap, data.selectedEdges );
        data.creases = mapEdges( *emap, data.creases );
    }
    else
    {
        data.mesh->topology.excludeLoneEdges( data.selectedEdges );
        data.mesh->topology.excludeLoneEdges( data.creases );
    }

    return res;
}

MRMESH_API std::optional<ObjectMeshData> makeDecimatedObjectMeshData( const ObjectMesh & obj, const DecimateSettings & settings,
    DecimateResult * outRes )
{
    MR_TIMER;

    ObjectMeshData data = obj.data();
    if ( !data.mesh )
    {
        assert( false );
        return {};
    }
    // clone mesh as well
    data.mesh = std::allocate_shared<Mesh>( Allocator<Mesh>(), *data.mesh );
    auto res = decimateObjectMeshData( data, settings );
    if ( outRes )
        *outRes = res;
    if ( res.cancelled )
        return {};
    return data;
}

bool remesh( MR::Mesh& mesh, const RemeshSettings & settings )
{
    MR_TIMER;
    if ( settings.progressCallback && !settings.progressCallback( 0.0f ) )
        return false;
    if ( settings.targetEdgeLen <= 0 )
    {
        assert( false );
        return false;
    }
    if ( settings.region && !settings.region->any() )
    {
        assert( false );
        return false;
    }

    SubdivideSettings subs;
    subs.maxEdgeLen = settings.targetEdgeLen;
    subs.maxEdgeSplits = settings.maxEdgeSplits;
    subs.maxAngleChangeAfterFlip = settings.maxAngleChangeAfterFlip;
    subs.maxDeviationAfterFlip = 0.1f * settings.targetEdgeLen; // the default 1 cuts off sharp edges of small meshes
    subs.subdivideBorder = !settings.frozenBoundary;
    subs.maxSplittableTriAspectRatio = settings.maxSplittableTriAspectRatio;
    subs.smoothMode = settings.useCurvature;
    subs.region = settings.region;
    subs.notFlippable = settings.notFlippable;
    subs.projectOnOriginalMesh = settings.projectOnOriginalMesh;
    subs.onEdgeSplit = settings.onEdgeSplit;
    subs.progressCallback = subprogress( settings.progressCallback, 0.0f, 0.5f );
    subdivideMesh( mesh, subs );
    if ( !reportProgress( settings.progressCallback, 0.5f ) )
        return false;

    // compute target number of triangles to get desired average edge length
    const auto regionArea = mesh.area( settings.region );
    const auto targetTriArea = sqr( settings.targetEdgeLen ) * ( sqrt( 3.0f ) / 4 ); // for equilateral triangle
    const auto targetNumTriD = regionArea / targetTriArea;
    const int targetNumTri = targetNumTriD < INT_MAX ? int( targetNumTriD ) : INT_MAX; // a tiny targetEdgeLen gives a ratio beyond int
    const auto currNumTri = settings.region ? (int)settings.region->count() : mesh.topology.numValidFaces();

    if ( currNumTri > targetNumTri )
    {
        DecimateSettings decs;
        decs.strategy = DecimateStrategy::ShortestEdgeFirst;
        decs.maxError = FLT_MAX;
        decs.maxEdgeLen = 1.5f * settings.targetEdgeLen; // not to over-decimate when there are many notFlippable edges in the region
        decs.maxDeletedFaces = currNumTri - targetNumTri;
        decs.touchBdVerts = !settings.frozenBoundary;
        decs.maxBdShift = settings.maxBdShift;
        decs.region = settings.region;
        decs.notFlippable = settings.notFlippable;
        decs.packMesh = settings.packMesh;
        decs.progressCallback = subprogress( settings.progressCallback, 0.5f, 0.95f );
        decs.preCollapse = settings.preCollapse;
        decs.onEdgeDel = settings.onEdgeDel;
        decs.stabilizer = 1e-6f;
        // it was a bad idea to make decs.stabilizer = settings.targetEdgeLen;
        // yes, it increased the uniformity of vertices, but shifted boundary vertices after edge collapse inside
        decimateMesh( mesh, decs );
        if ( !reportProgress( settings.progressCallback, 0.95f ) )
            return false;
    }

    if ( settings.finalRelaxIters > 0 )
    {
        // even if region is not given, we need to exclude mesh boundary from relaxation
        VertBitSet innerVerts = getInnerVerts( mesh.topology, settings.region );
        if ( settings.notFlippable )
            innerVerts -= getIncidentVerts( mesh.topology, *settings.notFlippable );
        MeshEqualizeTriAreasParams rp;
        rp.region = &innerVerts;
        rp.hardSmoothTetrahedrons = true;
        rp.noShrinkage = settings.finalRelaxNoShrinkage;
        DeloneSettings ds;
        ds.maxAngleChange = settings.maxAngleChangeAfterFlip;
        ds.region = settings.region;
        ds.notFlippable = settings.notFlippable;
        auto sp = subprogress( settings.progressCallback, 0.95f, 1.0f );
        for ( int i = 0; i < settings.finalRelaxIters; ++i )
        {
            if ( !reportProgress( sp, float( i ) / settings.finalRelaxIters ) )
                return false;
            equalizeTriAreas( mesh, rp );
            makeDeloneEdgeFlips( mesh, ds );
        }
    }

    return reportProgress( settings.progressCallback, 1.0f );
}

} //namespace MR
