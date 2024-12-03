#include "MRIntersectionContour.h"
#include "MRMesh.h"
#include "MRContoursCut.h"
#include "MRTimer.h"
#include "MRRegionBoundary.h"
#include "MRFillContour.h"
#include "MRAffineXf3.h"
#include "MRParallelFor.h"
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

bool isClosedContourTrivial( const MeshTopology& topology, const OneMeshContour& contour )
{
    assert( contour.closed );
    FaceBitSet fbs( topology.faceSize() );
    for ( const auto& inter : contour.intersections )
    {
        assert( inter.primitiveId.index() == OneMeshIntersection::Edge );
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

}

namespace MR
{

struct LinkedVET
{
    VariableEdgeTri vet;
    int index = -1; // index in initial array (needed to be able to find connections in parallel)
};

inline bool operator==( const LinkedVET& a, const LinkedVET& b )
{
    return a.vet == b.vet;
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

struct LinkedVETHash
{
    size_t operator()( const LinkedVET& lvet ) const
    {
        return ( ( 17 * lvet.vet.edge.undirected() + 23 * lvet.vet.tri ) << 1 ) + size_t( lvet.vet.isEdgeATriB );
    }
};

using LinkedVETSet = HashSet<LinkedVET, LinkedVETHash>;

struct AccumulativeSet
{
    const MeshTopology& topologyA;
    const MeshTopology& topologyB;

    LinkedVETSet hset;
    NeighborLinksList nListA; // flat list of neighbors filled in parallel
    NeighborLinksList nListB; // flat list of neighbors filled in parallel

    const MeshTopology& topologyByEdge( bool edgesATriB )
    {
        return edgesATriB ? topologyA : topologyB;
    }

    const MeshTopology& topologyByTri( bool edgesATriB )
    {
        return topologyByEdge( !edgesATriB );
    }

    bool empty() const
    {
        return hset.empty();
    }

    VariableEdgeTri getFirst() const
    {
        if ( !hset.empty() )
            return hset.begin()->vet;
        return {};
    }
};

LinkedVETSet createSet( const PreciseCollisionResult& intersections )
{
    LinkedVETSet set;
    set.reserve( ( intersections.edgesAtrisB.size() + intersections.edgesBtrisA.size() ) * 2 ); // 2 here is for mental peace
    for ( int i = 0; i < intersections.edgesAtrisB.size(); ++i )
        set.insert( { .vet = { intersections.edgesAtrisB[i],true },.index = i } );
    for ( int i = 0; i < intersections.edgesBtrisA.size(); ++i )
        set.insert( { .vet = { intersections.edgesBtrisA[i],false },.index = i } );
    return set;
}

const LinkedVET* find( const AccumulativeSet& accumulativeSet, const VariableEdgeTri& item )
{
    auto& itemSet = accumulativeSet.hset;
    auto it = itemSet.find( { item,-1 } );
    if ( it == itemSet.end() )
        return nullptr;
    return &( *it );
}

VariableEdgeTri orientBtoA( const VariableEdgeTri& curr )
{
    VariableEdgeTri res = curr;
    if ( !curr.isEdgeATriB )
        res.edge = res.edge.sym();
    return res;
}

const LinkedVET* findNext( AccumulativeSet& accumulativeSet, const VariableEdgeTri& curr )
{
    auto currB2Aedge = curr.isEdgeATriB ? curr.edge : curr.edge.sym();
    const auto& edgeTopology = accumulativeSet.topologyByEdge( curr.isEdgeATriB );
    const auto& triTopology = accumulativeSet.topologyByTri( curr.isEdgeATriB );
    auto leftTri = edgeTopology.left( currB2Aedge );
    auto leftEdge = triTopology.edgePerFace()[curr.tri];

    assert( curr.tri );

    if ( leftTri.valid() )
    {
        VariableEdgeTri variants[5] =
        {
            {{edgeTopology.next( currB2Aedge ),curr.tri},curr.isEdgeATriB},
            {{edgeTopology.prev( currB2Aedge.sym() ) ,curr.tri},curr.isEdgeATriB},

            {{leftEdge,leftTri},!curr.isEdgeATriB},
            {{triTopology.next( leftEdge ),leftTri},!curr.isEdgeATriB},
            {{triTopology.prev( leftEdge.sym() ),leftTri},!curr.isEdgeATriB}
        };

        for ( const auto& v : variants )
        {
            if ( !v.edge.valid() )
                continue;
            if ( auto found = find( accumulativeSet, v ) )
                return found;
        }
    }
    return nullptr;
}

void parallelPrepareLinkedLists( const PreciseCollisionResult& intersections, AccumulativeSet& accumulativeSet )
{
    MR_TIMER
    auto aSize = intersections.edgesAtrisB.size();
    accumulativeSet.nListA.resize( aSize );
    accumulativeSet.nListB.resize( intersections.edgesBtrisA.size() );
    ParallelFor( size_t( 0 ), aSize + accumulativeSet.nListB.size(), [&] ( size_t i )
    {
        bool eAtB = i < aSize;
        int aInd = int( i );
        int bInd = int( i - aSize );

        VariableEdgeTri curr = { eAtB ? intersections.edgesAtrisB[aInd] : intersections.edgesBtrisA[bInd] ,  eAtB };
        auto next = findNext( accumulativeSet, curr );
        if ( !next )
            return;
        auto& currItem = eAtB ? accumulativeSet.nListA[aInd] : accumulativeSet.nListB[bInd];
        auto& nextItem = next->vet.isEdgeATriB ? accumulativeSet.nListA[next->index] : accumulativeSet.nListB[next->index];
        currItem.next = int( next->vet.isEdgeATriB ? next->index : next->index + aSize );
        nextItem.prev = int( i );
    } );
}

ContinuousContours orderIntersectionContours( const AccumulativeSet& accumulativeSet, const PreciseCollisionResult& intersections )
{
    MR_TIMER
    struct CountourInfo
    {
        size_t startIndex;
        size_t size;
    };

    auto aSize = accumulativeSet.nListA.size();
    BitSet queuedRecords( aSize + accumulativeSet.nListB.size(), true );
    std::vector<CountourInfo> contInfos; // use it to preallocate contours and fill them in parallel then
    while ( queuedRecords.any() )
    {
        auto& currInfo = contInfos.emplace_back();
        
        currInfo.startIndex = queuedRecords.find_first();
        ++currInfo.size;
        bool closed = true;
        for ( auto nextIndex = currInfo.startIndex;; )
        {
            queuedRecords.reset( nextIndex );

            auto next = nextIndex < aSize ? accumulativeSet.nListA[nextIndex].next : accumulativeSet.nListB[nextIndex - aSize].next;
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
                const auto& prev = currInfo.startIndex < aSize ? 
                    accumulativeSet.nListA[currInfo.startIndex].prev : 
                    accumulativeSet.nListB[currInfo.startIndex - aSize].prev;
                if ( prev == -1 )
                    break;
                ++currInfo.size;
                currInfo.startIndex = prev;
                queuedRecords.reset( currInfo.startIndex );
            }
        }
    }

    ContinuousContours res( contInfos.size() );
    for ( int i = 0; i < res.size(); ++i )
    {
        res[i].resize( contInfos[i].size );
    }

    ParallelFor( res, [&] ( size_t i )
    {
        auto& resI = res[i];
        size_t index = contInfos[i].startIndex;
        for ( int j = 0; j < resI.size(); ++j )
        {
            auto curr = index < aSize ? intersections.edgesAtrisB[index] : intersections.edgesBtrisA[index - aSize];
            resI[j] = orientBtoA( { curr ,index < aSize } );
            index = index < aSize ? accumulativeSet.nListA[index].next : accumulativeSet.nListB[index - aSize].next;
        }
    } );

    return res;
}

ContinuousContours orderIntersectionContours( const MeshTopology& topologyA, const MeshTopology& topologyB, const PreciseCollisionResult& intersections )
{
    MR_TIMER
    AccumulativeSet accumulativeSet{ topologyA,topologyB, createSet( intersections ) };
    
    parallelPrepareLinkedLists( intersections, accumulativeSet );
    return orderIntersectionContours( accumulativeSet, intersections );
}

Contours3f extractIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& orientedContours, 
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    std::function<Vector3f( const Vector3f& coord, bool meshA )> getCoord;

    if ( !rigidB2A )
    {
        getCoord = [] ( const Vector3f& coord, bool )
        {
            return coord;
        };
    }
    else
    {
        getCoord = [xf = *rigidB2A] ( const Vector3f& coord, bool meshA )
        {
            return meshA ? coord : xf( coord );
        };
    }
    AffineXf3f inverseXf;
    if ( rigidB2A )
        inverseXf = rigidB2A->inverse();

    Contours3f res( orientedContours.size() );
    for ( int i = 0; i < res.size(); ++i )
    {
        const auto& inCont = orientedContours[i];
        auto& resI = res[i];
        resI.resize( inCont.size() );
        ParallelFor( inCont, [&] ( size_t j )
        {
            Vector3f a, b, c, d, e;
            const auto& vet = inCont[j];
            if ( vet.isEdgeATriB )
            {
                meshB.getTriPoints( vet.tri, a, b, c );
                d = meshA.orgPnt( vet.edge );
                e = meshA.destPnt( vet.edge );
            }
            else
            {
                meshA.getTriPoints( vet.tri, a, b, c );
                d = meshB.orgPnt( vet.edge );
                e = meshB.destPnt( vet.edge );
            }
            // always calculate in mesh A space
            resI[j] = findTriangleSegmentIntersectionPrecise(
                getCoord( a, !vet.isEdgeATriB ),
                getCoord( b, !vet.isEdgeATriB ),
                getCoord( c, !vet.isEdgeATriB ),
                getCoord( d, vet.isEdgeATriB ),
                getCoord( e, vet.isEdgeATriB ), converters );
        } );
    }
    return res;
}

std::vector<int> detectLoneContours( const ContinuousContours& contours )
{
    std::vector<int> res;
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
            res.push_back( i );
    }
    return res;
}

void removeLoneDegeneratedContours( const MeshTopology& edgesTopology, OneMeshContours& faceContours, OneMeshContours& edgeContours )
{
    for ( int i = int( faceContours.size() ) - 1; i >= 0; --i )
    {
        if ( faceContours[i].closed && calcLoneContourAreaSq( faceContours[i] ) == 0.0f && isClosedContourTrivial( edgesTopology, edgeContours[i] ) )
        {
            faceContours.erase( faceContours.begin() + i );
            edgeContours.erase( edgeContours.begin() + i );
        }
    }
}

void removeLoneContours( ContinuousContours& contours )
{
    auto loneContours = detectLoneContours( contours );
    for ( int i = int( loneContours.size() ) - 1; i >= 0; --i )
    {
        contours.erase( contours.begin() + loneContours[i] );
    }
}

} //namespace MR
