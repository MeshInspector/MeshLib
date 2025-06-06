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
        return 17 * size_t( vet.edge.undirected() ) + 23 * size_t( vet.tri );
    }
};

using EdgeTri2IndexMap = HashMap<EdgeTri, int, EdgeTriHash>;

using VariableEdgeTri2Index = std::pair<VariableEdgeTri, int>;

struct AccumulativeSet
{
    AccumulativeSet( const MeshTopology& topologyA, const MeshTopology& topologyB,
        const std::vector<EdgeTri>& edgesAtrisB, const std::vector<EdgeTri>& edgesBtrisA );

    const MeshTopology& topologyA;
    const MeshTopology& topologyB;

    EdgeTri2IndexMap edgeAtriBhmap, edgeBtriAhmap;
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
};

AccumulativeSet::AccumulativeSet( const MeshTopology& topologyA, const MeshTopology& topologyB,
    const std::vector<EdgeTri>& edgesAtrisB, const std::vector<EdgeTri>& edgesBtrisA )
    : topologyA( topologyA ), topologyB( topologyB )
{
    MR_TIMER;

    tbb::task_group taskGroup;
    taskGroup.run( [&] ()
    {
        edgeAtriBhmap.reserve( edgesAtrisB.size() );
        for ( int i = 0; i < edgesAtrisB.size(); ++i )
            edgeAtriBhmap[edgesAtrisB[i]] = i;
    } );

    edgeBtriAhmap.reserve( edgesBtrisA.size() );
    for ( int i = 0; i < edgesBtrisA.size(); ++i )
        edgeBtriAhmap[ edgesBtrisA[i] ] = i;
    taskGroup.wait();
}

const int* findIndex( const AccumulativeSet& accumulativeSet, const VariableEdgeTri& item )
{
    auto& itemSet = item.isEdgeATriB ? accumulativeSet.edgeAtriBhmap : accumulativeSet.edgeBtriAhmap;
    auto it = itemSet.find( item );
    if ( it == itemSet.end() )
        return {};
    return &it->second;
}

VariableEdgeTri orientBtoA( const VariableEdgeTri& curr )
{
    VariableEdgeTri res = curr;
    if ( !curr.isEdgeATriB )
        res.edge = res.edge.sym();
    return res;
}

std::optional<VariableEdgeTri2Index> findNext( AccumulativeSet& accumulativeSet, const VariableEdgeTri& curr )
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
            if ( auto pIndex = findIndex( accumulativeSet, v ) )
                return VariableEdgeTri2Index{ v, *pIndex };
        }
    }
    return {};
}

void parallelPrepareLinkedLists( const std::vector<EdgeTri>& edgesAtrisB, const std::vector<EdgeTri>& edgesBtrisA, AccumulativeSet& accumulativeSet )
{
    MR_TIMER;
    auto aSize = edgesAtrisB.size();
    accumulativeSet.nListA.resize( aSize );
    accumulativeSet.nListB.resize( edgesBtrisA.size() );
    ParallelFor( size_t( 0 ), aSize + accumulativeSet.nListB.size(), [&] ( size_t i )
    {
        bool eAtB = i < aSize;
        int aInd = int( i );
        int bInd = int( i - aSize );

        VariableEdgeTri curr = { eAtB ? edgesAtrisB[aInd] : edgesBtrisA[bInd] ,  eAtB };
        auto next = findNext( accumulativeSet, curr );
        if ( !next )
            return;
        auto& currItem = eAtB ? accumulativeSet.nListA[aInd] : accumulativeSet.nListB[bInd];
        auto& nextItem = next->first.isEdgeATriB ? accumulativeSet.nListA[next->second] : accumulativeSet.nListB[next->second];
        currItem.next = int( next->first.isEdgeATriB ? next->second : next->second + aSize );
        nextItem.prev = int( i );
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
    auto aSize = accumulativeSet.nListA.size();
    BitSet queuedRecords( aSize + accumulativeSet.nListB.size(), true );
    std::vector<ContourInfo> contInfos; // use it to preallocate contours and fill them in parallel then
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
    return contInfos;
}

ContinuousContours orderIntersectionContoursUsingAccumulativeSet( const AccumulativeSet& accumulativeSet, const std::vector<EdgeTri>& edgesAtrisB, const std::vector<EdgeTri>& edgesBtrisA )
{
    MR_TIMER;
    auto contInfos = calcContoursInfo( accumulativeSet );

    ContinuousContours res( contInfos.size() );
    for ( int i = 0; i < res.size(); ++i )
    {
        res[i].resize( contInfos[i].size );
    }

    auto aSize = accumulativeSet.nListA.size();
    ParallelFor( res, [&] ( size_t i )
    {
        auto& resI = res[i];
        size_t index = contInfos[i].startIndex;
        for ( int j = 0; j < resI.size(); ++j )
        {
            auto curr = index < aSize ? edgesAtrisB[index] : edgesBtrisA[index - aSize];
            resI[j] = orientBtoA( { curr ,index < aSize } );
            index = index < aSize ? accumulativeSet.nListA[index].next : accumulativeSet.nListB[index - aSize].next;
        }
    } );

    return res;
}

} //anonymous namespace

ContinuousContours orderSelfIntersectionContours( const MeshTopology& topology, const std::vector<EdgeTri>& intersections )
{
    MR_TIMER;
    AccumulativeSet accumulativeSet{ topology, topology, intersections,intersections };
    parallelPrepareLinkedLists( intersections, intersections, accumulativeSet );
    return orderIntersectionContoursUsingAccumulativeSet( accumulativeSet, intersections, intersections );
}

ContinuousContours orderIntersectionContours( const MeshTopology& topologyA, const MeshTopology& topologyB, const PreciseCollisionResult& intersections )
{
    MR_TIMER;
    AccumulativeSet accumulativeSet{ topologyA, topologyB, intersections.edgesAtrisB,intersections.edgesBtrisA };
    parallelPrepareLinkedLists( intersections.edgesAtrisB, intersections.edgesBtrisA, accumulativeSet );
    return orderIntersectionContoursUsingAccumulativeSet( accumulativeSet, intersections.edgesAtrisB, intersections.edgesBtrisA );
}

Contours3f extractIntersectionContours( const Mesh& meshA, const Mesh& meshB, const ContinuousContours& orientedContours,
    const CoordinateConverters& converters, const AffineXf3f* rigidB2A /*= nullptr */ )
{
    MR_TIMER;
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

bool isClosed( const ContinuousContour& contour )
{
    return contour.size() > 1 &&
        contour.front().isEdgeATriB == contour.back().isEdgeATriB &&
        contour.front().edge.undirected() == contour.back().edge.undirected() &&
        contour.front().tri == contour.back().tri;
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
