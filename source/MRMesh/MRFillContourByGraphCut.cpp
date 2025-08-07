#include "MRFillContourByGraphCut.h"
#include "MRMeshTopology.h"
#include "MREdgeIterator.h"
#include "MRRingIterator.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include <cfloat>
#include <deque>

namespace MR
{

static constexpr int Left = 0;
static constexpr int Right = 1;

static constexpr float ContourEdge = FLT_MAX;

class GraphCut
{
public:
    GraphCut( const MeshTopology & topology, const EdgeMetric & metric );
    void addContour( const EdgePath & contour );
    void addFaces( FaceBitSet source, FaceBitSet sink );
    FaceBitSet fill( const ProgressCallback& progress );

private:
    const MeshTopology & topology_;
    Vector<float, EdgeId> capacity_; // residual capacity of dual edge from left to right
    FaceBitSet filled_[2];
    Vector<EdgeId, FaceId> parent_;  // edge having parent to the right and this face to the left, invalid edge for root faces
    std::deque<FaceId> active_[2];
    std::vector<FaceId> orphans_;

    // process given active face which should belong to given side (left or right)
    void processActive_( FaceId f, int side );
    // augment the path joined at edge e
    void augment_( EdgeId e );
    // adapt orphans_ from given side
    void adapt_( int side );
    // tests whether grand is a grandparent of child
    bool isGrandparent_( FaceId child, FaceId grand ) const;
    // checks that there is not saturated path from f to a root
    bool checkNotSaturatedPath_( FaceId f, int side ) const;
};

GraphCut::GraphCut( const MeshTopology & topology, const EdgeMetric & metric ) : topology_( topology )
{
    MR_TIMER;

        auto szF = topology.lastValidFace() + 1;
    filled_[0].resize( szF );
    filled_[1].resize( szF );
    parent_.resize( szF );

    capacity_.resize( topology.edgeSize() );
    for ( EdgeId e : undirectedEdges( topology ) )
        capacity_[e] = capacity_[e.sym()] = metric( e );
}

void GraphCut::addContour( const EdgePath & contour )
{
    MR_TIMER;

    for ( auto e : contour )
        capacity_[e] = ContourEdge;

    for ( auto e : contour )
    {
        if ( capacity_[e.sym()] == ContourEdge )
            continue;
        capacity_[e.sym()] = ContourEdge;
        auto l = topology_.left( e );
        if ( l )
        {
            if ( !filled_[Left].test_set( l ) )
                active_[Left].push_back( l );
        }

        auto r = topology_.right( e );
        if ( r )
        {
            if ( !filled_[Right].test_set( r ) )
                active_[Right].push_back( r );
        }
    }

    auto bothLabels = filled_[Left] & filled_[Right];
    filled_[Left] -= bothLabels;
    filled_[Right] -= bothLabels;
}

void GraphCut::addFaces( FaceBitSet source, FaceBitSet sink )
{
    MR_TIMER;
    assert( !filled_[Left].intersects( filled_[Right] ) );

    // ignore faces marked simultaneously as source and sink
    const auto bothLabels = source & sink;
    source -= bothLabels;
    sink -= bothLabels;

    // do not change already filled faces
    const auto alreadyFilled = filled_[Left] | filled_[Right];
    source -= alreadyFilled;
    sink -= alreadyFilled;

    for ( auto f : source )
        active_[Left].push_back( f );

    for ( auto f : sink )
        active_[Right].push_back( f );

    filled_[Left] |= source;
    filled_[Right] |= sink;

    assert( !filled_[Left].intersects( filled_[Right] ) );
}

FaceBitSet GraphCut::fill( const ProgressCallback& progress )
{
    MR_TIMER;

    size_t numCycles = 0;
    while ( !active_[Left].empty() && !active_[Right].empty() )
    {
        auto lf = active_[Left].front();
        active_[Left].pop_front();
        processActive_( lf, Left );

        auto rf = active_[Right].front();
        active_[Right].pop_front();
        processActive_( rf, Right );

        ++numCycles;
        if ( !reportProgress( progress,
            [numCycles]() { return std::erf( numCycles * 1e-7f ); }, // report 84% for numCycles == 10'000'000
            numCycles, 65536 ) )
            break;
    }

    if ( active_[Right].empty() )
        return topology_.getValidFaces() - filled_[Right];

    return filled_[Left];
}

void GraphCut::processActive_( FaceId f, int side )
{
    if ( !filled_[side].test( f ) )
        return; // face has changed the side since the moment it was put in the queue
    assert( !filled_[1 - side].test( f ) );

    auto parent = parent_[f];
    assert( !parent || topology_.left( parent ) == f );

    for ( EdgeId e : leftRing( topology_, f ) )
    {
        if ( e == parent || capacity_[e] == ContourEdge )
            continue;
        auto r = topology_.right( e );
        if ( !r )
            continue;
        if ( filled_[1 - side].test( r ) )
        {
            augment_( side ? e.sym() : e );
            if ( !filled_[side].test( f ) )
                return; // face has changed the side during augmentation
        }
        else if ( !filled_[side].test( r ) && capacity_[side == Left ? e : e.sym()] > 0 )
        {
            filled_[side].set( r );
            parent_[r] = e.sym();
            assert( checkNotSaturatedPath_( r, side ) );
            active_[side].push_back( r );
        }
    }
}

void GraphCut::augment_( EdgeId e )
{
    auto l = topology_.left( e );
    auto r = topology_.right( e );
    assert( l && r );

    for ( ;; )
    {
        assert( filled_[Left].test( l ) );
        assert( !filled_[Left].test( r ) );
        assert( filled_[Right].test( r ) );
        assert( !filled_[Right].test( l ) );
        assert( checkNotSaturatedPath_( l, Left ) );
        assert( checkNotSaturatedPath_( r, Right ) );

        auto minResidualCapacity = capacity_[e];
        assert( minResidualCapacity >= 0 );
        if ( minResidualCapacity == 0 )
            break;

        for ( auto f = l;; )
        {
            auto parent = parent_[f];
            if ( !parent )
                break;
            assert( topology_.left( parent ) == f );
            minResidualCapacity = std::min( minResidualCapacity, capacity_[parent.sym()] );
            f = topology_.right( parent );
        }
        for ( auto f = r;; )
        {
            auto parent = parent_[f];
            if ( !parent )
                break;
            assert( topology_.left( parent ) == f );
            minResidualCapacity = std::min( minResidualCapacity, capacity_[parent] );
            f = topology_.right( parent );
        }

        assert( minResidualCapacity > 0 );
        capacity_[e] -= minResidualCapacity;
        capacity_[e.sym()] += minResidualCapacity;

        assert( orphans_.empty() );
        for ( auto f = l;; )
        {
            auto parent = parent_[f];
            if ( !parent )
                break;
            assert( topology_.left( parent ) == f );
            capacity_[parent] += minResidualCapacity;
            if ( ( capacity_[parent.sym()] -= minResidualCapacity ) == 0 )
            {
                orphans_.push_back( f );
                parent_[f] = EdgeId{};
            }
            f = topology_.right( parent );
        }
        adapt_( Left );

        assert( orphans_.empty() );
        for ( auto f = r;; )
        {
            auto parent = parent_[f];
            if ( !parent )
                break;
            assert( topology_.left( parent ) == f );
            capacity_[parent.sym()] += minResidualCapacity;
            if ( ( capacity_[parent] -= minResidualCapacity ) == 0 )
            {
                orphans_.push_back( f );
                parent_[f] = EdgeId{};
            }
            f = topology_.right( parent );
        }
        adapt_( Right );

        if ( !filled_[Left].test( l ) || !filled_[Right].test( r ) )
            break;
    }
}

void GraphCut::adapt_( int side )
{
    while ( !orphans_.empty() )
    {
        auto f = orphans_.back();
        orphans_.pop_back();
        if ( !filled_[side].test( f ) )
            continue;
        parent_[f] = EdgeId();
        for ( EdgeId e : leftRing( topology_, f ) )
        {
            auto r = topology_.right( e );
            if ( !r || !filled_[side].test( r ) )
                continue;
            auto cap = capacity_[side == Right ? e : e.sym()];
            if ( cap > 0 )
            {
                if ( isGrandparent_( r, f ) )
                    active_[side].push_front( r );
                else
                {
                    parent_[f] = e;
                    assert( checkNotSaturatedPath_( f, side ) );
                    break;
                }
            }
        }
        if ( !parent_[f] )
        {
            // parent has not been found
            filled_[side].reset( f );
            for ( EdgeId e : leftRing( topology_, f ) )
            {
                auto r = topology_.right( e );
                if ( !r )
                    continue;
                if ( e.sym() == parent_[r] )
                {
                    assert( filled_[side].test( r ) );
                    parent_[r] = EdgeId();
                    orphans_.push_back( r );
                }
                if ( filled_[1 - side].test( r ) )
                {
                    auto cap = capacity_[side == Left ? e : e.sym()];
                    if ( cap > 0 )
                        active_[1 - side].push_front( r );
                }
            }
        }
    }
}

bool GraphCut::isGrandparent_( FaceId f, FaceId grand ) const
{
    while ( f != grand )
    {
        auto parent = parent_[f];
        if ( !parent )
            return false;
        assert( topology_.left( parent ) == f );
        f = topology_.right( parent );
    }
    return true;
}

bool GraphCut::checkNotSaturatedPath_( FaceId f, int side ) const
{
    for ( ;; )
    {
        assert( filled_[side].test( f ) );
        assert( !filled_[1 - side].test( f ) );
        auto parent = parent_[f];
        if ( !parent )
            return true;
        assert( topology_.left( parent ) == f );
        if ( side == Left )
            assert( capacity_[parent.sym()] > 0 );
        else
            assert( capacity_[parent] > 0 );
        f = topology_.right( parent );
    }
}

FaceBitSet fillContourLeftByGraphCut( const MeshTopology & topology, const EdgePath & contour, const EdgeMetric & metric, const ProgressCallback& progress )
{
    MR_TIMER;
    GraphCut filler( topology, metric );
    filler.addContour( contour );
    return filler.fill( progress );
}

FaceBitSet fillContourLeftByGraphCut( const MeshTopology & topology, const std::vector<EdgePath> & contours, const EdgeMetric & metric, const ProgressCallback& progress )
{
    MR_TIMER;
    GraphCut filler( topology, metric );
    for ( auto & contour : contours )
        filler.addContour( contour );
    return filler.fill( progress );
}

FaceBitSet segmentByGraphCut( const MeshTopology& topology, const FaceBitSet& source, const FaceBitSet& sink, const EdgeMetric& metric, const ProgressCallback& progress )
{
    MR_TIMER;
    GraphCut filler( topology, metric );
    filler.addFaces( source, sink );
    return filler.fill( progress );
}

} // namespace MR
