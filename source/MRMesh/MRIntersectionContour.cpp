#include "MRIntersectionContour.h"
#include "MRMesh.h"
#include "MRContoursCut.h"
#include "MRTimer.h"
#include "MRRegionBoundary.h"
#include "MRFillContour.h"
#include "MRAffineXf3.h"
#include "MRParallelFor.h"
#include <parallel_hashmap/phmap.h>
#include <optional>

namespace MR
{

namespace
{

// returns 4 * (contour area)^2
float calcLoneContourAreaSq( const OneMeshContour& contour )
{
    Vector3f dblDirArea;
    for ( int i = 0; i + 1 < contour.intersections.size(); ++i )
        dblDirArea += cross( contour.intersections[i].coordinate, contour.intersections[i + 1].coordinate );
    return dblDirArea.lengthSq();
}

bool isClosedContourTrivial( const MeshTopology& topology, const OneMeshContour& contour )
{
    assert( contour.closed );
    FaceBitSet fbs( topology.faceSize() );
    for ( const auto& inter : contour.intersections )
    {
        assert( std::holds_alternative<EdgeId>( inter.primitiveId ) );
        auto eid = std::get<EdgeId>( inter.primitiveId );
        if ( auto l = topology.left( eid ) )
            fbs.set( l );
    }
    auto boundary = findLeftBoundary( topology, fbs );
    if ( boundary.empty() )
        return false;
    auto fillRes = fillContourLeft( topology, boundary.front() );
    return !fillRes.test( topology.right( boundary.front().front() ) );
}

// store data about connected intersections
// indexed flat: if edgeAtriB then index = index in edgeAtriB vector
//                  otherwise      index - (edgeAtriB vector).size() = index in edgeBtriA vector
struct NeighborLinks
{
    int prev{ -1 };
    int next{ -1 };
};

using NeighborLinksList = std::vector<NeighborLinks>;

struct EdgeTriHash
{
    size_t operator()( const EdgeTri& vet ) const
    {
        return 17 * size_t( vet.edge.undirected() >> 2 ) + 23 * size_t( vet.tri >> 2 );
    }
};

using VariableEdgeTri2Index = std::pair<VarEdgeTri, int>;

class EdgeTri2IndexMap
{
public:
    EdgeTri2IndexMap( const std::vector<VarEdgeTri>& intersections );

    const int* findIndex( const VarEdgeTri& item ) const;

private:
    static int bucket_( const VarEdgeTri& x )
    {
        return ( x.isEdgeATriB() << 4 ) |
               ( ( x.edge.undirected() & 3 ) << 2 ) |
               ( x.tri() & 3 );
    };

    using HashMapType = HashMap<EdgeTri, int, EdgeTriHash>;
    constexpr static inline int NumBuckets = 32;
    HashMapType hmaps_[NumBuckets];
};

EdgeTri2IndexMap::EdgeTri2IndexMap( const std::vector<VarEdgeTri>& intersections )
{
    MR_TIMER;
    int counts[NumBuckets] = {};

    for ( const auto & x : intersections )
        ++counts[bucket_( x )];

    ParallelFor( 0, NumBuckets, [&]( int ib )
    {
        HashMapType hmap;
        hmap.reserve( counts[ib] );
        for ( int i = 0; i < intersections.size(); ++i )
            if ( bucket_( intersections[i] ) == ib )
                hmap[intersections[i].edgeTri()] = i;
        hmaps_[ib] = std::move( hmap );
    } );
}

const int* EdgeTri2IndexMap::findIndex( const VarEdgeTri& item ) const
{
    auto& itemSet = hmaps_[bucket_( item )];
    auto it = itemSet.find( item.edgeTri() );
    if ( it == itemSet.end() )
        return {};
    return &it->second;
}

struct AccumulativeSet
{
    AccumulativeSet( const MeshTopology& topologyA, const MeshTopology& topologyB,
        const std::vector<VarEdgeTri>& intersections );

    const MeshTopology& topologyA;
    const MeshTopology& topologyB;

    NeighborLinksList nList; // flat indices of prev/next elements in the contour

    EdgeTri2IndexMap edgeTri2IndexMap; // map to flat index in nList

    const MeshTopology& topologyByEdge( bool edgesATriB )
    {
        return edgesATriB ? topologyA : topologyB;
    }

    const MeshTopology& topologyByTri( bool edgesATriB )
    {
        return topologyByEdge( !edgesATriB );
    }
};

AccumulativeSet::AccumulativeSet( const MeshTopology& topologyA, const MeshTopology& topologyB,
    const std::vector<VarEdgeTri>& intersections )
    : topologyA( topologyA ), topologyB( topologyB ), edgeTri2IndexMap( intersections )
{
}

inline VarEdgeTri orientBtoA( const VarEdgeTri& curr )
{
    VarEdgeTri res = curr;
    if ( !curr.isEdgeATriB() )
        res.edge = res.edge.sym();
    return res;
}

std::optional<VariableEdgeTri2Index> findNext( AccumulativeSet& accumulativeSet, const VarEdgeTri& curr )
{
    auto currB2Aedge = curr.isEdgeATriB() ? curr.edge : curr.edge.sym();
    const auto& edgeTopology = accumulativeSet.topologyByEdge( curr.isEdgeATriB() );
    const auto& triTopology = accumulativeSet.topologyByTri( curr.isEdgeATriB() );
    auto leftTri = edgeTopology.left( currB2Aedge );
    auto leftEdge = triTopology.edgePerFace()[curr.tri()];

    assert( curr.edge );

    if ( leftTri.valid() )
    {
        VarEdgeTri variants[5] =
        {
            { curr.isEdgeATriB(), edgeTopology.next( currB2Aedge ), curr.tri() },
            { curr.isEdgeATriB(), edgeTopology.prev( currB2Aedge.sym() ), curr.tri() },

            { !curr.isEdgeATriB(), leftEdge, leftTri },
            { !curr.isEdgeATriB(), triTopology.next( leftEdge ), leftTri },
            { !curr.isEdgeATriB(), triTopology.prev( leftEdge.sym() ), leftTri }
        };

        for ( const auto& v : variants )
        {
            if ( !v.edge.valid() )
                continue;
            if ( auto pIndex = accumulativeSet.edgeTri2IndexMap.findIndex( v ) )
                return VariableEdgeTri2Index{ v, *pIndex };
        }
    }
    return {};
}

void parallelPrepareLinkedLists( const std::vector<VarEdgeTri>& intersections, AccumulativeSet& accumulativeSet )
{
    MR_TIMER;
    const auto sz = (int)intersections.size();
    accumulativeSet.nList.resize( sz );
    ParallelFor( 0, sz, [&] ( int i )
    {
        const VarEdgeTri& curr = intersections[i];
        auto next = findNext( accumulativeSet, curr );
        if ( !next )
            return;
        auto& currItem = accumulativeSet.nList[i];
        auto& nextItem = accumulativeSet.nList[next->second];
        currItem.next = next->second;
        nextItem.prev = i;
    } );
}

struct ContourInfo
{
    size_t startIndex;
    size_t size;
};

std::vector<ContourInfo> calcContoursInfo( const AccumulativeSet& accumulativeSet )
{
    MR_TIMER;
    BitSet queuedRecords( accumulativeSet.nList.size(), true );
    std::vector<ContourInfo> contInfos; // use it to preallocate contours and fill them in parallel then
    for ( auto s : queuedRecords )
    {
        auto& currInfo = contInfos.emplace_back();

        currInfo.startIndex = s;
        ++currInfo.size;
        bool closed = true;
        for ( auto nextIndex = currInfo.startIndex;; )
        {
            queuedRecords.reset( nextIndex );

            auto next = accumulativeSet.nList[nextIndex].next;
            if ( next == -1 )
            {
                closed = false;
                break;
            }
            ++currInfo.size;
            nextIndex = next;
            if ( nextIndex == currInfo.startIndex )
                break;
        }

        if ( !closed )
        {
            for ( ;; )
            {
                const auto prev = accumulativeSet.nList[currInfo.startIndex].prev;
                if ( prev == -1 )
                    break;
                ++currInfo.size;
                currInfo.startIndex = prev;
                queuedRecords.reset( currInfo.startIndex );
            }
        }
    }
    return contInfos;
}

ContinuousContours orderIntersectionContoursUsingAccumulativeSet( const AccumulativeSet& accumulativeSet, const std::vector<VarEdgeTri>& intersections )
{
    MR_TIMER;
    const auto contInfos = calcContoursInfo( accumulativeSet );

    ContinuousContours res( contInfos.size() );
    ParallelFor( res, [&] ( size_t i )
    {
        const auto sz = contInfos[i].size;
        ContinuousContour resI;
        resI.reserve( sz );
        size_t index = contInfos[i].startIndex;
        for ( int j = 0; j < sz; ++j )
        {
            const auto& curr = intersections[index];
            resI.push_back( orientBtoA( curr ) );
            index = accumulativeSet.nList[index].next;
        }
        res[i] = std::move( resI );
    } );

    return res;
}

} //anonymous namespace

ContinuousContours orderSelfIntersectionContours( const MeshTopology& topology, const std::vector<EdgeTri>& intersections )
{
    MR_TIMER;
    std::vector<VarEdgeTri> vars;
    vars.reserve( intersections.size() * 2 );
    for ( const auto & x : intersections )
    {
        vars.emplace_back( true, x );
        vars.emplace_back( false, x );
    }
    AccumulativeSet accumulativeSet{ topology, topology, vars };
    parallelPrepareLinkedLists( vars, accumulativeSet );
    return orderIntersectionContoursUsingAccumulativeSet( accumulativeSet, vars );
}

ContinuousContours orderIntersectionContours( const MeshTopology& topologyA, const MeshTopology& topologyB, const PreciseCollisionResult& intersections )
{
    MR_TIMER;
    AccumulativeSet accumulativeSet{ topologyA, topologyB, intersections };
    parallelPrepareLinkedLists( intersections, accumulativeSet );
    return orderIntersectionContoursUsingAccumulativeSet( accumulativeSet, intersections );
}

Contours3f extractIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& orientedContours,
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    Contours3f res;
    getOneMeshIntersectionContours( meshA, meshB, orientedContours, nullptr, nullptr, converters, rigidB2A, &res );
    return res;
}

bool isClosed( const ContinuousContour& contour )
{
    return contour.size() > 1 &&
        contour.front() == contour.back();
}

std::vector<int> detectLoneContours( const ContinuousContours& contours, bool ignoreOpen )
{
    MR_TIMER;
    std::vector<int> res;
    for ( int i = 0; i < contours.size(); ++i )
    {
        auto& contour = contours[i];
        if ( contour.empty() || ( ignoreOpen && !isClosed( contour ) ) )
            continue;
        bool first = contour[0].isEdgeATriB();
        bool isLone = true;
        for ( const auto& vet : contour )
        {
            if ( vet.isEdgeATriB() != first )
            {
                isLone = false;
                break;
            }
        }
        if ( isLone )
            res.push_back( i );
    }
    return res;
}

void removeLoneDegeneratedContours( const MeshTopology& edgesTopology, OneMeshContours& faceContours, OneMeshContours& edgeContours )
{
    MR_TIMER;
    for ( int i = int( faceContours.size() ) - 1; i >= 0; --i )
    {
        if ( faceContours[i].closed && calcLoneContourAreaSq( faceContours[i] ) == 0.0f && isClosedContourTrivial( edgesTopology, edgeContours[i] ) )
        {
            faceContours.erase( faceContours.begin() + i );
            edgeContours.erase( edgeContours.begin() + i );
        }
    }
}

void removeLoneContours( ContinuousContours& contours, bool ignoreOpen )
{
    MR_TIMER;
    auto loneContours = detectLoneContours( contours, ignoreOpen );
    for ( int i = int( loneContours.size() ) - 1; i >= 0; --i )
    {
        contours.erase( contours.begin() + loneContours[i] );
    }
}

} //namespace MR
