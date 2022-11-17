#include "MRPolylineDecimate.h"
#include "MRQuadraticForm.h"
#include "MRBitSet.h"
#include "MRBitSetParallelFor.h"
#include "MRGTest.h"
#include "MRPolyline.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <queue>

namespace MR
{

// collapses given edge and
// 1) deletes vertex org( e )/dest( e ) if given edge was their only edge, otherwise only dest( e );
// 2) make edge e lone
// returns next( e ) if it is valid
EdgeId collapseEdge( PolylineTopology & topology, const EdgeId e )
{
    const EdgeId eNext = topology.next( e );
    if ( eNext == e )
    {
        topology.setOrg( e, VertId() );
        const EdgeId b = topology.next( e.sym() );
        if ( b == e.sym() )
            topology.setOrg( e.sym(), VertId() );
        else
            topology.splice( b, e.sym() );

        assert( topology.isLoneEdge( e ) );
        return EdgeId();
    }

    topology.splice( eNext, e );
    topology.setOrg( e.sym(), VertId() );

    const EdgeId a = topology.next( e.sym() );
    if ( a == e.sym() )
    {
        assert( topology.isLoneEdge( e ) );
        return eNext != e ? eNext : EdgeId();
    }

    topology.splice( a, e.sym() );
    topology.splice( eNext, a );

    assert( topology.isLoneEdge( e ) );
    return eNext != e ? eNext : EdgeId();
}

template<typename V>
class PolylineDecimator
{
public:
    PolylineDecimator( MR::Polyline<V> & polyline, const DecimatePolylineSettings<V> & settings );
    DecimatePolylineResult run();

    // returns true if the collapse of given edge is permitted by the region and settings
    bool isInRegion( EdgeId e ) const;

private: 
    MR::Polyline<V> & polyline_;
    const DecimatePolylineSettings<V> & settings_;
    const float maxErrorSq_;
    Vector<QuadraticForm<V>, VertId> vertForms_;
    struct QueueElement
    {
        float c = 0;
        UndirectedEdgeId uedgeId;
        std::pair<float, UndirectedEdgeId> asPair() const { return { -c, uedgeId }; }
        bool operator < ( const QueueElement & r ) const { return asPair() < r.asPair(); }
    };
    std::priority_queue<QueueElement> queue_;
    UndirectedEdgeBitSet presentInQueue_;
    DecimatePolylineResult res_;
    class EdgeMetricCalc;

    void initializeQueue_();
    std::optional<QueueElement> computeQueueElement_( UndirectedEdgeId ue, QuadraticForm<V> * outCollapseForm = nullptr, V * outCollapsePos = nullptr ) const;
    void addInQueueIfMissing_( UndirectedEdgeId ue );
    VertId collapse_( EdgeId edgeToCollapse, const V & collapsePos );
};

template<typename V>
PolylineDecimator<V>::PolylineDecimator( MR::Polyline<V> & polyline, const DecimatePolylineSettings<V> & settings )
    : polyline_( polyline )
    , settings_( settings )
    , maxErrorSq_( sqr( settings.maxError ) )
{
}

static bool isRegionEdge( const PolylineTopology & topology, EdgeId e, const VertBitSet * region )
{
    if ( !region )
        return true;
    return region->test( topology.org( e ) ) && region->test( topology.dest( e ) );
}

template<typename V>
bool PolylineDecimator<V>::isInRegion( EdgeId e ) const
{
    if ( !isRegionEdge( polyline_.topology, e, settings_.region ) )
        return false;
    if ( !settings_.touchBdVertices )
    {
        if ( polyline_.topology.next( e ) == e )
            return false;
        if ( polyline_.topology.next( e.sym() ) == e.sym() )
            return false;
    }
    return true;
}

template<typename V>
class PolylineDecimator<V>::EdgeMetricCalc 
{
public:
    EdgeMetricCalc( const PolylineDecimator<V> & decimator ) : decimator_( decimator ) { }
    EdgeMetricCalc( EdgeMetricCalc & x, tbb::split ) : decimator_( x.decimator_ ) { }
    void join( EdgeMetricCalc & y ) { auto yes = y.takeElements(); elems_.insert( elems_.end(), yes.begin(), yes.end() ); }

    const std::vector<QueueElement> & elements() const { return elems_; }
    std::vector<QueueElement> takeElements() { return std::move( elems_ ); }

    void operator()( const tbb::blocked_range<UndirectedEdgeId> & r ) 
    {
        const auto & polyline = decimator_.polyline_;
        for ( UndirectedEdgeId ue = r.begin(); ue < r.end(); ++ue ) 
        {
            EdgeId e{ ue };
            if ( polyline.topology.isLoneEdge( e ) )
                continue;
            if ( !decimator_.isInRegion( e ) )
                continue;
            if ( auto qe = decimator_.computeQueueElement_( ue ) )
                elems_.push_back( *qe );
        }
    }
            
public:
    const PolylineDecimator<V> & decimator_;
    std::vector<QueueElement> elems_;
};

template<typename V>
MR::QuadraticForm<V> computeFormAtVertex( const MR::Polyline<V> & polyline, MR::VertId v, float stabilizer )
{
    QuadraticForm<V> qf;
    
    const auto e = polyline.topology.edgeWithOrg( v );
    qf.addDistToLine( polyline.edgeVector( e ).normalized() );
    const auto eNext = polyline.topology.next( e );
    if ( e == eNext )
        stabilizer += 1;
    else
        qf.addDistToLine( polyline.edgeVector( eNext ).normalized() );

    qf.addDistToOrigin( stabilizer );
    return qf;
}

template<typename V>
void PolylineDecimator<V>::initializeQueue_()
{
    MR_TIMER;

    VertBitSet store;
    const VertBitSet & regionVertices = polyline_.topology.getVertIds( settings_.region );

    if ( settings_.vertForms && !settings_.vertForms->empty() )
    {
        assert( settings_.vertForms->size() > polyline_.topology.lastValidVert() );
        vertForms_ = std::move( *settings_.vertForms );
    }
    else
    {
        vertForms_.resize( polyline_.topology.lastValidVert() + 1 );
        BitSetParallelFor( regionVertices, [&]( VertId v )
        {
            vertForms_[v] = computeFormAtVertex( polyline_, v, settings_.stabilizer );
        } );
    }

    EdgeMetricCalc calc( *this );
    parallel_reduce( tbb::blocked_range<UndirectedEdgeId>( UndirectedEdgeId{0}, UndirectedEdgeId{polyline_.topology.undirectedEdgeSize()} ), calc );
    presentInQueue_.resize( polyline_.topology.undirectedEdgeSize() );
    for ( const auto & qe : calc.elements() )
        presentInQueue_.set( qe.uedgeId );
    queue_ = std::priority_queue<QueueElement>{ std::less<QueueElement>(), calc.takeElements() };
}

template<typename V>
auto PolylineDecimator<V>::computeQueueElement_( UndirectedEdgeId ue, QuadraticForm<V> * outCollapseForm, V * outCollapsePos ) const -> std::optional<QueueElement>
{
    EdgeId e{ ue };
    const auto o = polyline_.topology.org( e );
    const auto d = polyline_.topology.org( e.sym() );
    const auto po = polyline_.points[o];
    const auto pd = polyline_.points[d];
    auto [qf, pos] = sum( vertForms_[o], po, vertForms_[d], pd, !settings_.optimizeVertexPos );

    QueueElement res;
    res.uedgeId = ue;
    res.c = qf.c;

    if ( settings_.adjustCollapse )
    {
        const auto pos0 = pos;
        settings_.adjustCollapse( ue, res.c, pos );
        if ( res.c > maxErrorSq_ )
            return {};
        if ( outCollapseForm && pos != pos0 )
            qf.c = vertForms_[o].eval( po - pos ) + vertForms_[d].eval( pd - pos );
    }
    else if ( res.c > maxErrorSq_ )
        return {};

    if ( outCollapseForm )
        *outCollapseForm = qf;
    if ( outCollapsePos )
        *outCollapsePos = pos;
    return res;
}

template<typename V>
void PolylineDecimator<V>::addInQueueIfMissing_( UndirectedEdgeId ue )
{
    EdgeId e{ ue };
    if ( !isInRegion( e ) )
        return;
    if ( presentInQueue_.test_set( ue ) )
        return;
    if ( auto qe = computeQueueElement_( ue ) )
        queue_.push( *qe );
}

template<typename V>
VertId PolylineDecimator<V>::collapse_( EdgeId edgeToCollapse, const V & collapsePos )
{
    auto & topology = polyline_.topology;
    auto vo = topology.org( edgeToCollapse );
    if ( !settings_.optimizeVertexPos && collapsePos == polyline_.destPnt( edgeToCollapse ) )
    {
        // reverse the edge to have its origin in remaining fixed vertex
        edgeToCollapse = edgeToCollapse.sym();
        vo = topology.org( edgeToCollapse );
    }

    const auto o = polyline_.orgPnt( edgeToCollapse );
    const auto d = polyline_.destPnt( edgeToCollapse );
    float maxOldEdgeLenSq = std::max( sqr( settings_.maxEdgeLen ), ( o - d ).lengthSq() );
    float maxNewEdgeLenSq = 0;

    const auto ep = topology.next( edgeToCollapse ).sym();
    if ( ep != edgeToCollapse.sym() )
    {
        const auto epp = topology.next( ep ).sym();
        const auto eppp = topology.next( epp ).sym();
        if ( epp != ep.sym() && eppp != epp.sym() && eppp == edgeToCollapse )
            return {}; // keep at least a triangle from every closed contour

        const auto po = polyline_.orgPnt( ep );
        maxOldEdgeLenSq = std::max( maxOldEdgeLenSq, ( o - po ).lengthSq() );
        maxNewEdgeLenSq = std::max( maxNewEdgeLenSq, ( collapsePos - po ).lengthSq() );

        const auto en = topology.next( edgeToCollapse.sym() );
        if ( en != edgeToCollapse.sym() )
        {
            const auto nd = polyline_.destPnt( en );
            maxOldEdgeLenSq = std::max( maxOldEdgeLenSq, ( d - nd ).lengthSq() );
            maxNewEdgeLenSq = std::max( maxNewEdgeLenSq, ( collapsePos - nd ).lengthSq() );

            if ( dot( po - collapsePos, nd - collapsePos ) > 0 )
            {
                // after the collapse the line will make a sharp turn in remaining vertex
                if ( dot( po - o, d - o ) <= 0 && dot( o - d, nd - d ) <= 0 )
                    return {}; // there are no sharp turns before the collapse, so prohibit it
            }
        }
    }

    if ( maxNewEdgeLenSq > maxOldEdgeLenSq )
        return {}; // new edge would be longer than all of old edges and longer than allowed in settings

    if ( settings_.preCollapse && !settings_.preCollapse( edgeToCollapse, collapsePos ) )
        return {}; // user prohibits the collapse

    ++res_.vertsDeleted;

    polyline_.points[vo] = collapsePos;
    collapseEdge( topology, edgeToCollapse );
    return topology.hasVert( vo ) ? vo : VertId{};
}

template<typename V>
DecimatePolylineResult PolylineDecimator<V>::run()
{
    MR_TIMER;

    initializeQueue_();

    res_.errorIntroduced = settings_.maxError;
    while ( !queue_.empty() )
    {
        auto topQE = queue_.top();
        assert( presentInQueue_.test( topQE.uedgeId ) );
        queue_.pop();
        if( res_.vertsDeleted >= settings_.maxDeletedVertices )
        {
            res_.errorIntroduced = std::sqrt( topQE.c );
            break;
        }

        if ( polyline_.topology.isLoneEdge( topQE.uedgeId ) )
        {
            // edge has been deleted by this moment
            presentInQueue_.reset( topQE.uedgeId );
            continue;
        }

        QuadraticForm<V> collapseForm;
        V collapsePos;
        auto qe = computeQueueElement_( topQE.uedgeId, &collapseForm, &collapsePos );
        if ( !qe )
        {
            presentInQueue_.reset( topQE.uedgeId );
            continue;
        }

        if ( qe->c > topQE.c )
        {
            queue_.push( *qe );
            continue;
        }

        presentInQueue_.reset( topQE.uedgeId );
        VertId collapseVert = collapse_( topQE.uedgeId, collapsePos );
        if ( !collapseVert )
            continue;

        vertForms_[collapseVert] = collapseForm;

        EdgeId e = polyline_.topology.edgeWithOrg( collapseVert );
        addInQueueIfMissing_( e.undirected() );
        EdgeId eNext = polyline_.topology.next( e );
        if ( e != eNext )
            addInQueueIfMissing_( eNext.undirected() );
    }

    if ( settings_.vertForms )
        *settings_.vertForms = std::move( vertForms_ );
    return res_;
}

DecimatePolylineResult decimatePolyline( MR::Polyline2 & polyline, const DecimatePolylineSettings2 & settings )
{
    MR_TIMER;
    //MR_POLYLINE_WRITER( polyline );
    PolylineDecimator<Vector2f> md( polyline, settings );
    return md.run();
}

DecimatePolylineResult decimatePolyline( MR::Polyline3 & polyline, const DecimatePolylineSettings3 & settings )
{
    MR_TIMER;
    //MR_POLYLINE_WRITER( polyline );
    PolylineDecimator<Vector3f> md( polyline, settings );
    return md.run();
}

DecimatePolylineResult decimateContour( Contour2f& contour, const DecimatePolylineSettings2& settings )
{
    MR_TIMER;
    MR::Polyline2 p( { contour } );
    auto res = decimatePolyline( p, settings );
    auto resContours = p.contours();
    assert ( p.contours().size() == 1 );
    if ( !p.contours().empty() )
        contour = p.contours()[0];
    else
        contour.clear();
    return res;
}

DecimatePolylineResult decimateContour( Contour3f& contour, const DecimatePolylineSettings3& settings )
{
    MR_TIMER;
    MR::Polyline3 p( { contour } );
    auto res = decimatePolyline( p, settings );
    auto resContours = p.contours();
    assert ( p.contours().size() == 1 );
    if ( !p.contours().empty() )
        contour = p.contours()[0];
    else
        contour.clear();
    return res;
}

TEST( MRMesh, DecimatePolyline )
{
    std::vector< Contour2f> testContours;
    // rhombus
    Contour2f contRhombus;
    contRhombus.push_back( Vector2f( 0.f, 0.f ) );
    contRhombus.push_back( Vector2f( 1.f, 1.f ) );
    contRhombus.push_back( Vector2f( 2.f, 1.f ) );
    contRhombus.push_back( Vector2f( 2.f, 0.f ) );
    contRhombus.push_back( Vector2f( 0.f, 0.f ) );
    testContours.push_back( contRhombus );

    // square
    Contour2f contSquare;
    contRhombus.push_back( Vector2f( 0.f, 0.f ) );
    contRhombus.push_back( Vector2f( 0.f, 1.f ) );
    contRhombus.push_back( Vector2f( 1.f, 1.f ) );
    contRhombus.push_back( Vector2f( 1.f, 0.f ) );
    testContours.push_back( contRhombus );

    // square with self-intersections
    Contour2f contSquareSelfIntersected;
    contSquareSelfIntersected.push_back( Vector2f( 0.f, 0.f ) );
    contSquareSelfIntersected.push_back( Vector2f( 1.f, 1.f ) );
    contSquareSelfIntersected.push_back( Vector2f( 0.f, 1.f ) );
    contSquareSelfIntersected.push_back( Vector2f( 1.f, 0.f ) );
    contSquareSelfIntersected.push_back( Vector2f( 0.f, 0.f ) );
    testContours.push_back( contSquareSelfIntersected );

    // simple small line
    Contour2f smallLine;
    smallLine.push_back( Vector2f( 0.f, 0.f ) );
    smallLine.push_back( Vector2f( 1.f, 1.f ) );
    smallLine.push_back( Vector2f( 2.f, 2.f ) );
    testContours.push_back( smallLine );

    // arc
    Contour2f contArc;
    contArc.push_back( Vector2f( -2.f, 0.f ) );
    contArc.push_back( Vector2f( -1.f, 1.f ) );
    contArc.push_back( Vector2f( 0.f, 1.5f ) );
    contArc.push_back( Vector2f( 1.f, 1.f ) );
    contArc.push_back( Vector2f( 2.f, 0.f ) );
    testContours.push_back( contArc );

    for( auto& cont : testContours )
    {
        DecimatePolylineSettings2 settings;
        settings.maxDeletedVertices = 3;
        settings.maxError = 100.f;
        settings.touchBdVertices = false;

        MR::Polyline2 pl( { cont } );
        auto plBack = pl;
        auto decRes = decimatePolyline( pl, settings );

        int validLines = 0;
        for ( UndirectedEdgeId ue{0}; ue < pl.topology.undirectedEdgeSize(); ++ue )
            if ( !pl.topology.isLoneEdge( ue ) )
                ++validLines;

        EXPECT_TRUE( validLines > 0 );
        EXPECT_EQ( validLines + decRes.vertsDeleted, plBack.topology.undirectedEdgeSize() );
    }
}

} //namespace MR
