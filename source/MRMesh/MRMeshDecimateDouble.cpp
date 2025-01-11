#include "MRMeshDecimateDouble.h"
#include "MRMesh.h"
#include "MRQuadraticForm.h"
#include "MRRegionBoundary.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRRingIterator.h"
#include "MRTriMath.h"
#include "MRTimer.h"
#include "MRCylinder.h"
#include "MRGTest.h"
#include "MRLineSegm.h"
#include <queue>

namespace MR
{

using VertCoordsD = Vector<Vector3d, VertId>;

class MeshDeciamatorDouble
{
public:
    MeshDeciamatorDouble( Mesh & mesh, const DecimateSettingsDouble & settings );
    DecimateResult run( Mesh & mesh );

    const Vector3d& orgPnt( EdgeId e ) const { return points_[topology_.org( e )]; }
    const Vector3d& destPnt( EdgeId e ) const { return points_[topology_.dest( e )]; }
    LineSegm3d edgeSegment( EdgeId e ) const { return { orgPnt( e ), destPnt( e ) }; }
    double edgeLengthSq( EdgeId e ) const { return edgeSegment( e ).lengthSq(); }

private:
    MeshTopology & topology_;
    VertCoordsD points_;
    const DecimateSettingsDouble & settings_;
    const double maxErrorSq_;
    Vector<QuadraticForm3d, VertId> myVertForms_;
    Vector<QuadraticForm3d, VertId> * pVertForms_ = nullptr;
    UndirectedEdgeBitSet regionEdges_;
    VertBitSet myBdVerts_;
    const VertBitSet * pBdVerts_ = nullptr;
    std::function<void( EdgeId del, EdgeId rem )> onEdgeDel_;

    enum class EdgeOp : unsigned int
    {
        CollapseOptPos, ///< collapse the edge with target position optimization
        CollapseEnd,    ///< collapse the edge in one of its current vertices
        Flip            ///< flip the edge inside quadrangle
        // one more option is available to fit in 2 bits
    };

    struct QueueElement
    {
        double c = 0;
        struct X {
            EdgeOp edgeOp : 2 = EdgeOp::CollapseOptPos;
            unsigned int uedgeId : 30 = 0;
        } x;
        UndirectedEdgeId uedgeId() const { return UndirectedEdgeId{ (int)x.uedgeId }; }
        std::pair<double, int> asPair() const { return { -c, x.uedgeId }; }
        bool operator < ( const QueueElement & r ) const { return asPair() < r.asPair(); }
    };
    static_assert( sizeof( QueueElement ) == 16 );
    std::priority_queue<QueueElement> queue_;
    UndirectedEdgeBitSet presentInQueue_;
    DecimateResult res_;
    std::vector<VertId> originNeis_;
    std::vector<Vector3d> triDblAreas_; // directed double areas of newly formed triangles to check that they are consistently oriented
    class EdgeMetricCalc;

    bool initializeQueue_();
    QuadraticForm3d collapseForm_( UndirectedEdgeId ue, const Vector3d & collapsePos ) const;
    std::optional<QueueElement> computeQueueElement_( UndirectedEdgeId ue, bool optimizeVertexPos,
        QuadraticForm3d * outCollapseForm = nullptr, Vector3d * outCollapsePos = nullptr ) const;
    void addInQueueIfMissing_( UndirectedEdgeId ue );
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
    CanCollapseRes canCollapse_( EdgeId edgeToCollapse, const Vector3d & collapsePos ); // not const because it changes temporary originNeis_ and triDblAreas_

    /// performs edge collapse after previous successful check by canCollapse_
    /// \return org( edgeToCollapse ) or invalid id if it was the last edge
    VertId forceCollapse_( EdgeId edgeToCollapse, const Vector3d & collapsePos );

    struct CollapseRes
    {
        /// if collapse failed (or it was the last edge) then it is invalid, otherwise it is the remaining vertex
        VertId v;
        CollapseStatus status = CollapseStatus::Ok;
    };
    /// canCollapse_ + forceCollapse_
    CollapseRes collapse_( EdgeId edgeToCollapse, const Vector3d & collapsePos );
};

MeshDeciamatorDouble::MeshDeciamatorDouble( Mesh & mesh, const DecimateSettingsDouble & settings )
    : topology_( mesh.topology )
    , points_( begin( mesh.points ), end( mesh.points ) )
    , settings_( settings )
    , maxErrorSq_( sqr( settings.maxError ) )
{
    if ( settings_.notFlippable || settings_.edgesToCollapse || settings_.twinMap || settings_.onEdgeDel )
    {
        onEdgeDel_ =
        [
            notFlippable = settings_.notFlippable,
            edgesToCollapse = settings_.edgesToCollapse,
            twinMap = settings_.twinMap,
            onEdgeDel = settings_.onEdgeDel
        ] ( EdgeId del, EdgeId rem )
        {
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
}

class MeshDeciamatorDouble::EdgeMetricCalc 
{
public:
    EdgeMetricCalc( const MeshDeciamatorDouble & decimator ) : decimator_( decimator ) { }
    EdgeMetricCalc( EdgeMetricCalc & x, tbb::split ) : decimator_( x.decimator_ ) { }
    void join( EdgeMetricCalc & y ) { auto yes = y.takeElements(); elems_.insert( elems_.end(), yes.begin(), yes.end() ); }

    const std::vector<QueueElement> & elements() const { return elems_; }
    std::vector<QueueElement> takeElements() { return std::move( elems_ ); }

    void operator()( const tbb::blocked_range<UndirectedEdgeId> & r ) 
    {
        const bool optimizeVertexPos = decimator_.settings_.optimizeVertexPos;
        for ( UndirectedEdgeId ue = r.begin(); ue < r.end(); ++ue ) 
        {
            EdgeId e{ ue };
            if ( decimator_.regionEdges_.empty() )
            {
                if ( decimator_.topology_.isLoneEdge( e ) )
                    continue;
            }
            else
            {
                if ( !decimator_.regionEdges_.test( ue ) )
                    continue;
            }
            if ( auto qe = decimator_.computeQueueElement_( ue, optimizeVertexPos ) )
                elems_.push_back( *qe );
        }
    }

public:
    const MeshDeciamatorDouble & decimator_;
    std::vector<QueueElement> elems_;
};

QuadraticForm3d computeFormAtVertex( const MeshTopology & topology, const VertCoordsD & points, const FaceBitSet * region,
    MR::VertId v, double stabilizer, bool angleWeigted, const UndirectedEdgeBitSet * creases )
{
    QuadraticForm3d qf;
    auto edgeVector = [&]( EdgeId e )
    {
        return points[topology.dest( e )] - points[topology.org( e )];
    };
    auto leftNormal = [&]( EdgeId e )
    {
        VertId a, b, c;
        topology.getLeftTriVerts( e, a, b, c );
        assert( a.valid() && b.valid() && c.valid() );
        const auto & ap = points[a];
        const auto & bp = points[b];
        const auto & cp = points[c];
        return cross( bp - ap, cp - ap ).normalized();
    };
    for ( EdgeId e : orgRing( topology, v ) )
    {
        if ( topology.isBdEdge( e, region ) || ( creases && creases->test( e ) ) )
        {
            // zero-length boundary edge is treated as uniform stabilizer: all shift directions are equally penalized,
            // otherwise it penalizes the shift proportionally to the distance from the line containing the edge
            qf.addDistToLine( edgeVector( e ).normalized() );
        }
        if ( topology.left( e ) ) // intentionally do not check that left face is in region to respect its plane as well
        {
            if ( angleWeigted )
            {
                auto d0 = edgeVector( e );
                auto d1 = edgeVector( topology.next( e ) );
                auto angle = MR::angle( d0, d1 );
                static constexpr float INV_PIF = 1 / PI_F;
                qf.addDistToPlane( leftNormal( e ), angle * INV_PIF );
            }
            else
            {
                // zero-area triangle is treated as no triangle with no penalty at all,
                // otherwise it penalizes the shift proportionally to the distance from the plane containing the triangle
                qf.addDistToPlane( leftNormal( e ) );
            }
        }
    }
    qf.addDistToOrigin( stabilizer );
    return qf;
}

Vector<QuadraticForm3d, VertId> computeFormsAtVertices( const MeshTopology & topology, const VertCoordsD & points, const FaceBitSet * region,
    double stabilizer, bool angleWeigted, const UndirectedEdgeBitSet * creases )
{
    MR_TIMER;

    VertBitSet store;
    const VertBitSet & regionVertices = getIncidentVerts( topology, region, store );

    Vector<QuadraticForm3d, VertId> res( regionVertices.find_last() + 1 );
    BitSetParallelFor( regionVertices, [&]( VertId v )
    {
        res[v] = computeFormAtVertex( topology, points, region, v, stabilizer, angleWeigted, creases );
    } );

    return res;
}

bool MeshDeciamatorDouble::initializeQueue_()
{
    MR_TIMER;

    if ( settings_.vertForms )
        pVertForms_ = settings_.vertForms;
    else
        pVertForms_ = &myVertForms_;

    if ( pVertForms_->empty() )
        *pVertForms_ = computeFormsAtVertices( topology_, points_, settings_.region, settings_.stabilizer, settings_.angleWeightedDistToPlane, settings_.notFlippable );

    if ( settings_.progressCallback && !settings_.progressCallback( 0.1f ) )
        return false;

    // initialize regionEdges_ if some edges (out-of-region or touching boundary) cannot be collapsed
    if ( settings_.region )
    {
        // all region edges
        regionEdges_ = getIncidentEdges( topology_, *settings_.region );
        if ( settings_.edgesToCollapse )
            regionEdges_ &= *settings_.edgesToCollapse;
        if ( !settings_.touchNearBdEdges )
        {
            // exclude edges touching boundary
            BitSetParallelFor( regionEdges_, [&]( UndirectedEdgeId ue )
            {
                if ( pBdVerts_->test( topology_.org( ue ) ) ||
                     pBdVerts_->test( topology_.dest( ue ) ) )
                    regionEdges_.reset( ue );
            } );
        }
    }
    else if ( !settings_.touchNearBdEdges )
    {
        assert( !settings_.region );
        regionEdges_.clear();
        regionEdges_.resize( topology_.undirectedEdgeSize(), true );
        // exclude lone edges and edges touching boundary
        BitSetParallelForAll( regionEdges_, [&]( UndirectedEdgeId ue )
        {
            if ( ( settings_.edgesToCollapse && !settings_.edgesToCollapse->test( ue ) ) ||
                 topology_.isLoneEdge( ue ) ||
                 pBdVerts_->test( topology_.org( ue ) ) ||
                 pBdVerts_->test( topology_.dest( ue ) ) )
                regionEdges_.reset( ue );
        } );
    }
    else if ( settings_.edgesToCollapse )
        regionEdges_ = *settings_.edgesToCollapse;

    EdgeMetricCalc calc( *this );
    parallel_reduce( tbb::blocked_range<UndirectedEdgeId>( UndirectedEdgeId{0}, UndirectedEdgeId{topology_.undirectedEdgeSize()} ), calc );

    if ( settings_.progressCallback && !settings_.progressCallback( 0.2f ) )
        return false;

    presentInQueue_.resize( topology_.undirectedEdgeSize() );
    for ( const auto & qe : calc.elements() )
        presentInQueue_.set( qe.uedgeId() );
    queue_ = std::priority_queue<QueueElement>{ std::less<QueueElement>(), calc.takeElements() };

    if ( settings_.progressCallback && !settings_.progressCallback( 0.25f ) )
        return false;
    return true;
}

QuadraticForm3d MeshDeciamatorDouble::collapseForm_( UndirectedEdgeId ue, const Vector3d & collapsePos ) const
{
    EdgeId e{ ue };
    const auto o = topology_.org( e );
    const auto d = topology_.dest( e );
    const auto po = points_[o];
    const auto pd = points_[d];
    const auto vo = (*pVertForms_)[o];
    const auto vd = (*pVertForms_)[d];
    return sumAt( vo, po, vd, pd, collapsePos );
}

auto MeshDeciamatorDouble::computeQueueElement_( UndirectedEdgeId ue, bool optimizeVertexPos,
    QuadraticForm3d * outCollapseForm, Vector3d * outCollapsePos ) const -> std::optional<QueueElement>
{
    EdgeId e{ ue };
    const auto o = topology_.org( e );
    const auto d = topology_.dest( e );
    const auto po = points_[o];
    const auto pd = points_[d];
    const auto vo = (*pVertForms_)[o];
    const auto vd = (*pVertForms_)[d];

    std::optional<QueueElement> res;
    // prepares res; checks flip metric; returns true if the edge does not collpase and function can return
    auto earlyReturn = [&]( double errSq )
    {
        EdgeOp edgeOp = optimizeVertexPos ? EdgeOp::CollapseOptPos : EdgeOp::CollapseEnd;
        if ( settings_.maxAngleChange >= 0 && ( !settings_.notFlippable || !settings_.notFlippable->test( ue ) ) )
        {
            assert( false );
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
        if ( earlyReturn( edgeLengthSq( e ) ) )
            return res;
    }

    QuadraticForm3d qf;
    Vector3d pos;
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
                qf.c = DBL_MAX;
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

void MeshDeciamatorDouble::addInQueueIfMissing_( UndirectedEdgeId ue )
{
    if ( !regionEdges_.empty() && !regionEdges_.test( ue ) )
        return;
    if ( presentInQueue_.test( ue ) )
        return;
    if ( auto qe = computeQueueElement_( ue, settings_.optimizeVertexPos ) )
    {
        queue_.push( *qe );
        presentInQueue_.set( ue );
    }
}

void MeshDeciamatorDouble::flipEdge_( UndirectedEdgeId ue )
{
    EdgeId e = ue;
    topology_.flipEdge( e );
    assert( topology_.left( e ) );
    assert( topology_.right( e ) );
    addInQueueIfMissing_( e.undirected() );
    addInQueueIfMissing_( topology_.prev( e ).undirected() );
    addInQueueIfMissing_( topology_.next( e ).undirected() );
    addInQueueIfMissing_( topology_.prev( e.sym() ).undirected() );
    addInQueueIfMissing_( topology_.next( e.sym() ).undirected() );
}

auto MeshDeciamatorDouble::canCollapse_( EdgeId edgeToCollapse, const Vector3d & collapsePos ) -> CanCollapseRes
{
    const auto & topology = topology_;
    auto vl = topology.left( edgeToCollapse ).valid()  ? topology.dest( topology.next( edgeToCollapse ) ) : VertId{};
    auto vr = topology.right( edgeToCollapse ).valid() ? topology.dest( topology.prev( edgeToCollapse ) ) : VertId{};

    // cannot collapse internal edge if its left and right faces share another edge
    if ( vl && vr )
    {
        if ( auto pe = topology.prev( edgeToCollapse ); pe != edgeToCollapse && pe == topology.next( edgeToCollapse ) )
            return { .status =  CollapseStatus::SharedEdge };
        if ( auto pe = topology.prev( edgeToCollapse.sym() ); pe != edgeToCollapse.sym() && pe == topology.next( edgeToCollapse.sym() ) )
            return { .status =  CollapseStatus::SharedEdge };
    }
    const bool collapsingFlippable = !settings_.notFlippable || !settings_.notFlippable->test( edgeToCollapse );

    auto vo = topology.org( edgeToCollapse );
    auto vd = topology.dest( edgeToCollapse );
    auto po = points_[vo];
    auto pd = points_[vd];
    if ( collapsePos == pd )
    {
        // reverse the edge to have its origin in remaining fixed vertex
        edgeToCollapse = edgeToCollapse.sym();
        std::swap( vo, vd );
        std::swap( vl, vr );
        std::swap( po, pd );
    }

    auto smallShift = [maxBdShiftSq = sqr( settings_.maxBdShift )]( const LineSegm3d & segm, const Vector3d & p )
    {
        return ( closestPointOnLineSegm( p, segm ) - p ).lengthSq() <= maxBdShiftSq;
    };
    if ( ( !vl || !vr ) && settings_.maxBdShift < DBL_MAX )
    {
        if ( !smallShift( edgeSegment( edgeToCollapse ), collapsePos ) )
            return { .status =  CollapseStatus::PosFarBd }; // new vertex is too far from collapsing boundary edge
        if ( !vr )
        {
            if ( !smallShift( LineSegm3d{ orgPnt( topology_.prevLeftBd( edgeToCollapse ) ), collapsePos }, po ) )
                return { .status =  CollapseStatus::PosFarBd }; // origin of collapsing boundary edge is too far from new boundary segment
            if ( !smallShift( LineSegm3d{ destPnt( topology_.nextLeftBd( edgeToCollapse ) ), collapsePos }, pd ) )
                return { .status =  CollapseStatus::PosFarBd }; // destination of collapsing boundary edge is too far from new boundary segment
        }
        if ( !vl )
        {
            if ( !smallShift( LineSegm3d{ orgPnt( topology_.prevLeftBd( edgeToCollapse.sym() ) ), collapsePos }, pd ) )
                return { .status =  CollapseStatus::PosFarBd }; // destination of collapsing boundary edge is too far from new boundary segment
            if ( !smallShift( LineSegm3d{ destPnt( topology_.nextLeftBd( edgeToCollapse.sym() ) ), collapsePos }, po ) )
                return { .status =  CollapseStatus::PosFarBd }; // origin of collapsing boundary edge is too far from new boundary segment
        }
    }

    double maxOldAspectRatio = settings_.maxTriangleAspectRatio;
    double maxNewAspectRatio = 0;
    const double edgeLenSq = ( po - pd ).lengthSq();
    double maxOldEdgeLenSq = std::max( sqr( settings_.maxEdgeLen ), edgeLenSq );
    double maxNewEdgeLenSq = 0;

    bool normalFlip = false; // at least one triangle flips its normal or a degenerate triangle becomes not-degenerate
    originNeis_.clear();
    triDblAreas_.clear();
    Vector3d sumDblArea_;
    EdgeId oBdEdge; // a boundary edge !right(e) incident to org( edgeToCollapse )
    for ( EdgeId e : orgRing0( topology, edgeToCollapse ) )
    {
        const auto eDest = topology.dest( e );
        if ( eDest == vd )
            return { .status =  CollapseStatus::MultipleEdge }; // multiple edge found
        if ( eDest != vl && eDest != vr )
            originNeis_.push_back( eDest );

        const auto pDest = points_[eDest];
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

        const auto pDest2 = destPnt( topology.next( e ) );
        if ( eDest != vr )
        {
            auto da = cross( pDest - collapsePos, pDest2 - collapsePos );
            if ( !normalFlip )
            {
                const auto oldA = cross( pDest - po, pDest2 - po );
                if ( dot( da, oldA ) <= 0 )
                    normalFlip = true;
            }
            triDblAreas_.push_back( da );
            sumDblArea_ += Vector3d{ da };
            const auto triAspect = triangleAspectRatio( collapsePos, pDest, pDest2 );
            if ( triAspect >= settings_.criticalTriAspectRatio )
                triDblAreas_.back() = Vector3d{}; //cannot trust direction of degenerate triangles
            maxNewAspectRatio = std::max( maxNewAspectRatio, triAspect );
        }
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( po, pDest, pDest2 ) );
    }
    if ( oBdEdge
        && !smallShift( LineSegm3d{ po, destPnt( oBdEdge ) }, collapsePos )
        && !smallShift( LineSegm3d{ po, orgPnt( topology.prevLeftBd( oBdEdge ) ) }, collapsePos ) )
            return { .status =  CollapseStatus::PosFarBd }; // new vertex is too far from both existed boundary edges
    std::sort( originNeis_.begin(), originNeis_.end() );

    EdgeId dBdEdge; // a boundary edge !right(e) incident to dest( edgeToCollapse )
    for ( EdgeId e : orgRing0( topology, edgeToCollapse.sym() ) )
    {
        const auto eDest = topology.dest( e );
        assert ( eDest != vo );
        if ( std::binary_search( originNeis_.begin(), originNeis_.end(), eDest ) )
            return { .status =  CollapseStatus::MultipleEdge }; // to prevent appearance of multiple edges

        const auto pDest = points_[eDest];
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

        const auto pDest2 = destPnt( topology.next( e ) );
        if ( eDest != vl )
        {
            auto da = cross( pDest - collapsePos, pDest2 - collapsePos );
            if ( !normalFlip )
            {
                const auto oldA = cross( pDest - pd, pDest2 - pd );
                if ( dot( da, oldA ) <= 0 )
                    normalFlip = true;
            }
            triDblAreas_.push_back( da );
            sumDblArea_ += Vector3d{ da };
            const auto triAspect = triangleAspectRatio( collapsePos, pDest, pDest2 );
            if ( triAspect >= settings_.criticalTriAspectRatio )
                triDblAreas_.back() = Vector3d{}; //cannot trust direction of degenerate triangles
            maxNewAspectRatio = std::max( maxNewAspectRatio, triAspect );
        }
        maxOldAspectRatio = std::max( maxOldAspectRatio, triangleAspectRatio( pd, pDest, pDest2 ) );
    }
    if ( dBdEdge
        && !smallShift( LineSegm3d{ pd, destPnt( dBdEdge ) }, collapsePos )
        && !smallShift( LineSegm3d{ pd, orgPnt( topology.prevLeftBd( dBdEdge ) ) }, collapsePos ) )
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

    // if at least one triangle normal flips, checks that all new normals are consistent
    if ( normalFlip && ( ( po != pd ) || ( po != collapsePos ) ) )
    {
        auto n = Vector3d{ sumDblArea_.normalized() };
        for ( const auto da : triDblAreas_ )
            if ( dot( da, n ) < 0 )
                return { .status =  CollapseStatus::NormalFlip };
    }

    if ( settings_.preCollapse && !settings_.preCollapse( edgeToCollapse, collapsePos ) )
        return { .status =  CollapseStatus::User }; // user prohibits the collapse

    assert( topology.org( edgeToCollapse ) == vo );
    return { .e = edgeToCollapse };
}

VertId MeshDeciamatorDouble::forceCollapse_( EdgeId edgeToCollapse, const Vector3d & collapsePos )
{
    ++res_.vertsDeleted;

    auto & topology = topology_;
    const auto l = topology.left( edgeToCollapse );
    const auto r = topology.left( edgeToCollapse.sym() );
    if ( l )
        ++res_.facesDeleted;
    if ( r )
        ++res_.facesDeleted;

    const auto vo = topology.org( edgeToCollapse );
    points_[vo] = collapsePos;
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

    // update edges around remaining vertex
    for ( EdgeId e : orgRing( topology_, vo ) )
    {
        addInQueueIfMissing_( e.undirected() );
        if ( topology_.left( e ) )
            addInQueueIfMissing_( topology_.prev( e.sym() ).undirected() );
    }
    return vo;
}

auto MeshDeciamatorDouble::collapse_( EdgeId edgeToCollapse, const Vector3d & collapsePos ) -> CollapseRes
{
    CanCollapseRes can = canCollapse_( edgeToCollapse, collapsePos );
    if ( can.status != CollapseStatus::Ok )
        return { .status = can.status };
    assert( can.e == edgeToCollapse || can.e == edgeToCollapse.sym() );
    return { .v = forceCollapse_( can.e, collapsePos ) };
}

static void optionalPackMesh( Mesh & mesh, const DecimateSettingsDouble & settings )
{
    if ( !settings.packMesh )
        return;

    MR_TIMER
    FaceMap fmap;
    VertMap vmap;
    WholeEdgeMap emap;
    mesh.pack(
        settings.region ? &fmap : nullptr,
        settings.vertForms ? &vmap : nullptr,
        settings.notFlippable ? &emap : nullptr );

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

    if ( settings.notFlippable )
        *settings.notFlippable = settings.notFlippable->getMapping( [&emap]( UndirectedEdgeId i ) { return emap[i].undirected(); }, mesh.topology.undirectedEdgeSize() );
}

DecimateResult MeshDeciamatorDouble::run( Mesh & mesh )
{
    MR_TIMER
    assert( &topology_ == &mesh.topology );

    if ( settings_.bdVerts )
        pBdVerts_ = settings_.bdVerts;
    else
    {
        pBdVerts_ = &myBdVerts_;
        if ( !settings_.touchNearBdEdges )
            myBdVerts_ = getBoundaryVerts( topology_, settings_.region );
    }

    if ( !initializeQueue_() )
        return res_;

    res_.errorIntroduced = (float)settings_.maxError;
    int lastProgressFacesDeleted = 0;
    const int maxFacesDeleted = std::min(
        settings_.region ? (int)settings_.region->count() : topology_.numValidFaces(), settings_.maxDeletedFaces );
    while ( !queue_.empty() )
    {
        const auto topQE = queue_.top();
        const auto ue = topQE.uedgeId();
        assert( presentInQueue_.test( ue ) );
        queue_.pop();
        if ( res_.facesDeleted >= settings_.maxDeletedFaces || res_.vertsDeleted >= settings_.maxDeletedVertices )
        {
            res_.errorIntroduced = (float)std::sqrt( topQE.c );
            break;
        }

        if ( settings_.progressCallback && res_.facesDeleted >= 1000 + lastProgressFacesDeleted ) 
        {
            if ( !settings_.progressCallback( 0.25f + 0.75f * res_.facesDeleted / maxFacesDeleted ) )
                return res_;
            lastProgressFacesDeleted = res_.facesDeleted;
        }

        if ( topology_.isLoneEdge( ue ) )
        {
            // edge has been deleted by this moment
            presentInQueue_.reset( ue );
            continue;
        }

        QuadraticForm3d collapseForm;
        Vector3d collapsePos;
        auto qe = computeQueueElement_( ue, topQE.x.edgeOp == EdgeOp::CollapseOptPos, &collapseForm, &collapsePos );
        if ( !qe )
        {
            presentInQueue_.reset( ue );
            continue;
        }

        if ( qe->c > topQE.c )
        {
            queue_.push( *qe );
            continue;
        }

        presentInQueue_.reset( ue );

        UndirectedEdgeId twin;
        if ( settings_.twinMap )
            twin = getAt( *settings_.twinMap, ue );

        if ( qe->x.edgeOp == EdgeOp::Flip )
        {
            flipEdge_( ue );
            if ( twin )
                flipEdge_( twin );
        }
        else
        {
            // edge collapse
            const auto canCollapseRes = canCollapse_( ue, collapsePos );
            if ( canCollapseRes.status != CollapseStatus::Ok )
            {
                if ( topQE.x.edgeOp == EdgeOp::CollapseOptPos && geomFail_( canCollapseRes.status ) )
                {
                    qe = computeQueueElement_( ue, false );
                    if ( qe )
                    {
                        queue_.push( *qe );
                        presentInQueue_.set( ue );
                    }
                }
                continue;
            }
            if ( twin )
            {
                const auto twinCollapseForm = collapseForm_( twin, collapsePos );
                const auto twinCollapseRes = collapse_( twin, collapsePos );
                if ( twinCollapseRes.status != CollapseStatus::Ok )
                    continue;
                if ( twinCollapseRes.v )
                    (*pVertForms_)[twinCollapseRes.v] = twinCollapseForm;
            }
            if ( auto v = forceCollapse_( canCollapseRes.e, collapsePos ) )
                (*pVertForms_)[v] = collapseForm;
        }
    }

    // write points back in mesh
    BitSetParallelFor( topology_.getValidVerts(), [&]( VertId v )
    {
        mesh.points[v] = Vector3f( points_[v] );
    } );

    if ( settings_.progressCallback && !settings_.progressCallback( 1.0f ) )
        return res_;

    optionalPackMesh( mesh, settings_ );
    res_.cancelled = false;
    return res_;
}

static DecimateResult decimateMeshSerial( Mesh & mesh, const DecimateSettingsDouble & settings )
{
    MR_TIMER
    if ( settings.maxDeletedFaces <= 0 || settings.maxDeletedVertices <= 0 )
    {
        DecimateResult res;
        res.cancelled = false;
        return res;
    }
    MR_WRITER( mesh );
    MeshDeciamatorDouble md( mesh, settings );
    return md.run( mesh );
}

DecimateResult decimateMeshDouble( Mesh & mesh, const DecimateSettingsDouble & settings0 )
{
    auto settings = settings0;
    if ( settings.maxError >= DBL_MAX &&
         settings.maxEdgeLen >= DBL_MAX &&
         settings.maxDeletedFaces >= INT_MAX &&
         settings.maxDeletedVertices >= INT_MAX )
    {
        assert( false ); // below logic can be deleted in future
        // if unexperienced user started decimation with default settings, then decimate half of faces
        settings.maxDeletedFaces = int( settings.region ? settings.region->count() : mesh.topology.numValidFaces() ) / 2;
    }

    assert( settings.subdivideParts == 1 );
    return decimateMeshSerial( mesh, settings );
}

// check if Decimator updates region
TEST( MRMesh, MeshDecimateDouble )
{
    Mesh meshCylinder = makeCylinderAdvanced(0.5f, 0.5f, 0.0f, 20.0f / 180.0f * PI_F, 1.0f, 16);

    // select all faces
    MR::FaceBitSet regionForDecimation = meshCylinder.topology.getValidFaces();
    MR::FaceBitSet regionSaved(regionForDecimation);

    // setup and run decimator
    DecimateSettingsDouble decimateSettings;
    decimateSettings.maxError = 0.001;
    decimateSettings.region = &regionForDecimation;
    decimateSettings.maxTriangleAspectRatio = 80.0;

    auto decimateResults = decimateMeshDouble(meshCylinder, decimateSettings);

    // compare regions and deleted vertices and faces
    ASSERT_NE(regionSaved, regionForDecimation);
    ASSERT_GT(decimateResults.vertsDeleted, 0);
    ASSERT_GT(decimateResults.facesDeleted, 0);
}

} //namespace MR
