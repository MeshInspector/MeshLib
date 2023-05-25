#include "MRIntersectionContour.h"
#include "MRMeshTopology.h"
#include "MRContoursCut.h"
#include "MRTimer.h"
#include "MRRegionBoundary.h"
#include "MRFillContour.h"
#include <parallel_hashmap/phmap.h>

namespace
{

using namespace MR;
// returns 4 * (contour area)^2
float calcLoneContourAreaSq( const OneMeshContour& contour )
{
    Vector3f dblDirArea;
    for ( int i = 0; i + 1 < contour.intersections.size(); ++i )
        dblDirArea += cross( contour.intersections[i].coordinate, contour.intersections[i + 1].coordinate );
    return dblDirArea.lengthSq();
}

bool isLoneContourSingular( const OneMeshContour& contour )
{
    if ( contour.intersections.size() < 2 )
        return true;
    const auto& first = contour.intersections.front();
    for ( int i = 1; i < contour.intersections.size(); ++i )
    {
        if ( contour.intersections[i].coordinate != first.coordinate )
            return false;
    }
    return true;
}

bool isClosedContourTrivial( const MeshTopology& topology, const ContinuousContour& contour )
{
    FaceBitSet fbs( topology.faceSize() );
    for ( const auto& inter : contour )
    {
        auto eid = inter.edge;
        if ( auto l = topology.left( eid ) )
            fbs.set( l );
    }
    auto boundary = findLeftBoundary( topology, fbs );
    if ( boundary.empty() )
        return false;
    auto fillRes = fillContourLeft( topology, boundary.front() );
    return !fillRes.test( topology.right( boundary.front().front() ) );
}

}

namespace MR
{

struct EdgeTriHash
{
    size_t operator()( const EdgeTri& et ) const
    {
        return 17 * et.edge.undirected() + 23 * et.tri;
    }
};

using EdgeTriSet = phmap::flat_hash_set<EdgeTri, EdgeTriHash>;

struct AccumulativeSet
{
    const MeshTopology& topologyA;
    const MeshTopology& topologyB;

    EdgeTriSet eAtB;
    EdgeTriSet eBtA;

    const MeshTopology& topologyByEdge( bool edgesATriB )
    {
        return edgesATriB ? topologyA : topologyB;
    }

    const MeshTopology& topologyByTri( bool edgesATriB )
    {
        return topologyByEdge( !edgesATriB );
    }

    EdgeTriSet& set( bool edgesATriB )
    {
        return edgesATriB ? eAtB : eBtA;
    }
    const EdgeTriSet& set( bool edgesATriB ) const
    {
        return edgesATriB ? eAtB : eBtA;
    }
    bool empty() const
    {
        return eAtB.empty() && eBtA.empty();
    }
    VariableEdgeTri getFirst() const
    {
        if ( !eAtB.empty() )
            return {*eAtB.begin(),true};
        else if ( !eBtA.empty() )
            return {*eBtA.begin(),false};
        return {};
    }
};

EdgeTriSet createSet( const std::vector<EdgeTri>& edgeTris )
{
    EdgeTriSet set;
    set.reserve( edgeTris.size() * 2 ); // 2 here is for mental peace
    for ( const auto& edgeTri : edgeTris )
        set.insert( edgeTri );
    return set;
}

bool erase( AccumulativeSet& accumulativeSet, VariableEdgeTri& item )
{
    auto& itemSet = accumulativeSet.set( item.isEdgeATriB );
    auto it = itemSet.find( item );
    if ( it == itemSet.end() )
        return false;
    item = {*it,item.isEdgeATriB};
    itemSet.erase( it );
    return true;
}

VariableEdgeTri orientBtoA( const VariableEdgeTri& curr )
{
    VariableEdgeTri res = curr;
    if ( !curr.isEdgeATriB )
        res.edge = res.edge.sym();
    return res;
}

VariableEdgeTri sym( const VariableEdgeTri& curr )
{
    VariableEdgeTri res = curr;
    res.edge = res.edge.sym();
    return res;
}

bool getNext( AccumulativeSet& accumulativeSet, const VariableEdgeTri& curr, VariableEdgeTri& next )
{
    const auto& edgeTopology = accumulativeSet.topologyByEdge( curr.isEdgeATriB );
    const auto& triTopology = accumulativeSet.topologyByTri( curr.isEdgeATriB );
    auto leftTri = edgeTopology.left( curr.edge );
    auto leftEdge = triTopology.edgePerFace()[curr.tri];

    assert( curr.tri );

    if ( leftTri.valid() )
    {
        VariableEdgeTri variants[5] =
        {
            {{edgeTopology.next( curr.edge ),curr.tri},curr.isEdgeATriB},
            {{edgeTopology.prev( curr.edge.sym() ) ,curr.tri},curr.isEdgeATriB},

            {{leftEdge,leftTri},!curr.isEdgeATriB},
            {{triTopology.next( leftEdge ),leftTri},!curr.isEdgeATriB},
            {{triTopology.prev( leftEdge.sym() ),leftTri},!curr.isEdgeATriB}
        };

        for ( const auto& v : variants )
        {
            if ( !v.edge.valid() )
                continue;
            next = v;
            if ( erase( accumulativeSet, next ) )
                return true;
        }
    }
    return false;
}

ContinuousContour orderFirstIntersectionContour( AccumulativeSet& accumulativeSet )
{
    ContinuousContour forwardRes;
    auto first = accumulativeSet.getFirst();
    forwardRes.push_back( orientBtoA( first ) );
    VariableEdgeTri next;
    while ( getNext( accumulativeSet, forwardRes.back(), next ) )
    {
        forwardRes.push_back( orientBtoA( next ) );
    }

    // if not closed
    if ( erase( accumulativeSet, first ) )
    {
        ContinuousContour backwardRes;
        backwardRes.push_back( orientBtoA( first ) );
        while ( getNext( accumulativeSet, sym( backwardRes.back() ), next ) )
        {
            backwardRes.push_back( orientBtoA( next ) );
        }
        forwardRes.insert( forwardRes.begin(), backwardRes.rbegin(), backwardRes.rend() - 1 );
    }
    return forwardRes;
}

ContinuousContours orderIntersectionContours( const MeshTopology& topologyA, const MeshTopology& topologyB, const PreciseCollisionResult& intersections )
{
    MR_TIMER;
    AccumulativeSet accumulativeSet{topologyA,topologyB, createSet( intersections.edgesAtrisB ),createSet( intersections.edgesBtrisA ),};
    ContinuousContours res;
    while ( !accumulativeSet.empty() )
    {
        res.push_back( orderFirstIntersectionContour( accumulativeSet ) );
    }
    return res;
}

BitSet detectLoneContours( const ContinuousContours& contours )
{
    BitSet res( contours.size() );
    for ( int i = 0; i < contours.size(); ++i )
    {
        auto& contour = contours[i];
        if ( contour.empty() )
            continue;
        bool first = contour[0].isEdgeATriB;
        bool isLone = true;
        for ( const auto& vet : contour )
        {
            if ( vet.isEdgeATriB != first )
            {
                isLone = false;
                break;
            }
        }
        if ( isLone )
            res.set( i );
    }
    return res;
}

BitSet detectSingularContours( const OneMeshContours& faceContours, const BitSet& contourIds )
{
    BitSet res( contourIds.size() );
    for ( auto i : contourIds )
    {
        if ( isLoneContourSingular( faceContours[i] ) )
            res.set( i );
    }
    return res;
}

BitSet detectDegeneratedContours( const OneMeshContours& faceContours, const BitSet& contourIds )
{
    BitSet res( contourIds.size() );
    for ( auto i : contourIds )
    {
        if ( calcLoneContourAreaSq( faceContours[i] ) == 0.0f )
            res.set( i );
    }
    return res;
}

BitSet detectNonTrivialContours( const MeshTopology& tpA, const MeshTopology& tpB, const ContinuousContours& contours, const BitSet& contourIds )
{
    BitSet res( contourIds.size() );
    for ( auto i : contourIds )
    {
        if ( contours[i].empty() )
            continue;
        bool edgeATriB = contours[i].front().isEdgeATriB;
        if ( isClosedContourTrivial( edgeATriB ? tpA : tpB, contours[i] ) )
            res.set( i );
    }
    return res;
}

void removeContours( OneMeshContours& faceContours, const BitSet& contourIds )
{
    int counter = 0;
    for ( auto i : contourIds )
    {
        faceContours.erase( faceContours.begin() + i - counter );
        ++counter;
    }
}

void removeContours( ContinuousContours& contours, const BitSet& contourIds )
{
    int counter = 0;
    for ( auto i : contourIds )
    {
        contours.erase( contours.begin() + i - counter );
        ++counter;
    }
}

} //namespace MR
