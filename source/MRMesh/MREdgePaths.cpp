#include "MREdgePaths.h"
#include "MRMesh.h"
#include "MREdgeIterator.h"
#include "MRRingIterator.h"
#include "MRBitSet.h"
#include "MRRegionBoundary.h"
#include "MRphmap.h"
#include "MRPlane3.h"
#include "MRTimer.h"
#include "MRCube.h"
#include "MRUnionFind.h"
#include "MRGTest.h"
#include <queue>

namespace MR
{

bool isEdgePath( const MeshTopology & topology, const std::vector<EdgeId> & edges )
{
    for ( int i = 0; i + 1 < edges.size(); ++i )
        if ( topology.org( edges[i + 1] ) != topology.dest( edges[i] ) )
            return false;
    return true;
}

bool isEdgeLoop( const MeshTopology & topology, const std::vector<EdgeId> & edges )
{
    return isEdgePath( topology, edges ) && topology.org( edges.front() ) == topology.dest( edges.back() );
}

void reverse( EdgePath & path )
{
    std::reverse( path.begin(), path.end() );
    for ( auto & e : path )
        e = e.sym();
}

void reverse( std::vector<EdgePath> & paths )
{
    for ( auto & path : paths )
        reverse( path );
}

double calcPathMetric( const EdgePath & path, EdgeMetric metric )
{
    double res = 0;
    for ( auto & e : path )
        res += metric( e );
    return res;
}

void sortPathsByMetric( std::vector<EdgePath> & paths, EdgeMetric metric )
{
    MR_TIMER
    const auto sz = paths.size();
    std::vector<int> sortedIds( sz );
    std::vector<double> lens( sz );
    for ( int i = 0; i < sz; ++i )
    {
        sortedIds[i] = i;
        lens[i] = calcPathMetric( paths[i], metric );
    }
    std::sort( sortedIds.begin(), sortedIds.end(), [&]( int a, int b )
        { return lens[a] < lens[b]; } );

    std::vector<EdgePath> sorted( sz );
    for ( int i = 0; i < sz; ++i )
    {
        sorted[i] = std::move( paths[sortedIds[i]] );
    }
    paths = std::move( sorted );
}

void addLeftBand( const MeshTopology & topology, const EdgeLoop & loop, FaceBitSet & addHere )
{
    if ( loop.empty() )
        return;
    assert( isEdgeLoop( topology, loop ) );

    EdgeId stop = loop.back().sym();
    for ( EdgeId e : loop )
    {
        for ( EdgeId ei : orgRing( topology, e ) )
        {
            if ( ei == stop )
                break;
            if ( auto l = topology.left( ei ) )
                addHere.autoResizeSet( l );
        }
        stop = e.sym();
    }
}

EdgeMetric identityMetric() 
{ 
    return []( EdgeId ) { return 1.0f; }; 
}

EdgeMetric edgeLengthMetric( const Mesh & mesh )
{
    return [&mesh]( EdgeId e )
    {
        return mesh.edgeLength( e );
    };
}

EdgeMetric edgeCurvMetric( const Mesh & mesh, float angleSinFactor, float angleSinForBoundary )
{
    const float bdFactor = exp( angleSinFactor * angleSinForBoundary );

    return [&mesh, angleSinFactor, bdFactor ]( EdgeId e ) -> float
    {
        auto edgeLen = mesh.edgeLength( e );
        if ( mesh.topology.isBdEdge( e, nullptr ) )
            return edgeLen * bdFactor;

        return edgeLen * exp( angleSinFactor * mesh.dihedralAngleSin( e ) );
    };
}

EdgeMetric edgeTableMetric( const MeshTopology & topology, const EdgeMetric & metric )
{
    MR_TIMER

    Vector<float, UndirectedEdgeId> table( topology.undirectedEdgeSize() );
    for ( auto e : undirectedEdges( topology ) )
        table[e] = metric( e );

    return [table = std::move( table )]( EdgeId e )
    {
        return table[e.undirected()];
    };
}

struct VertPathInfo
{
    // edge from this vertex to its predecessor in the forest
    EdgeId back;
    // best summed metric to reach this vertex
    float metric = FLT_MAX;

    bool isStart() const { return !back.valid(); }
};

// smaller metric to be the first
inline bool operator <( const VertPathInfo & a, const VertPathInfo & b )
{
    return a.metric > b.metric;
}

using VertPathInfoMap = ParallelHashMap<VertId, VertPathInfo>;

class EdgePathsBuilder
{
public:
    EdgePathsBuilder( const MeshTopology & topology, const EdgeMetric & metric );
    // registers start vertex for paths
    void addStart( EdgeId edgeFromStart, float startMetric );
    // registers start region for paths, only boundary vertices are actually added and only outside steps
    void addStartRegion( const VertBitSet & region, float startMetric );
    // include one more edge in the edge forest, returning vertex-info for the newly reached vertex
    VertPathInfo growOneEdge();
    // whether new candidates are added in growOneEdge (true), or only old candidates are returned (false)
    bool keepGrowing() const { return keepGrowing_; }
    void stopGrowing() { keepGrowing_ = false; }

public:
    // returns true if further edge forest growth is impossible
    bool done() const { return nextSteps_.empty(); }
    // returns path length till the next candidate vertex or maximum float value if all vertices have been reached
    float doneDistance() const { return nextSteps_.empty() ? FLT_MAX : nextSteps_.top().metric; }
    // gives read access to the map from vertex to path to it
    const VertPathInfoMap & vertPathInfoMap() const { return vertPathInfoMap_; }
    // returns one element from the map (or nullptr if the element is missing)
    const VertPathInfo * getVertInfo( VertId v ) const;

    // returns the path in the forest from given vertex to one of start vertices
    std::vector<EdgeId> getPathBack( VertId backpathStart ) const;

private:
    const MeshTopology & topology_;
    EdgeMetric metric_;
    VertPathInfoMap vertPathInfoMap_;
    std::priority_queue<VertPathInfo> nextSteps_;
    bool keepGrowing_ = true;

    // compares proposed step with the value known for org( c.back );
    // if proposed step is smaller then adds it in the queue and returns true;
    // otherwise if the known metric to org( c.back ) is already not greater than returns false
    bool addNextStep_( const VertPathInfo & c );
    // adds steps for all origin ring edges of org( back ) including back itself and exluding skipRegion vertices;
    // returns true if at least one step was added
    bool addOrgRingSteps_( float orgMetric, EdgeId back, const VertBitSet * skipRegion = nullptr );
};

EdgePathsBuilder::EdgePathsBuilder( const MeshTopology & topology, const EdgeMetric & metric )
    : topology_( topology )
    , metric_( metric )
{
}

void EdgePathsBuilder::addStart( EdgeId edgeFromStart, float startMetric )
{
    auto & vi = vertPathInfoMap_[topology_.org( edgeFromStart )];
    if ( vi.metric <= startMetric )
        return;
    addOrgRingSteps_( vi.metric = startMetric, edgeFromStart );
}

void EdgePathsBuilder::addStartRegion( const VertBitSet & region, float startMetric )
{
    MR_TIMER
    for ( auto v : region )
    {
        if ( addOrgRingSteps_( startMetric, topology_.edgeWithOrg( v ), &region ) )
        {
            auto & vi = vertPathInfoMap_[v];
            assert ( vi.metric > startMetric );
            vi.metric = startMetric;
        }
    }
}

const VertPathInfo * EdgePathsBuilder::getVertInfo( VertId v ) const
{
    auto it = vertPathInfoMap_.find( v );
    return ( it != vertPathInfoMap_.end() ) ? &it->second : nullptr;
}

std::vector<EdgeId> EdgePathsBuilder::getPathBack( VertId v ) const
{
    std::vector<EdgeId> res;
    for (;;)
    {
        auto it = vertPathInfoMap_.find( v );
        if ( it == vertPathInfoMap_.end() )
        {
            assert( false );
            break;
        }
        auto & vi = it->second;
        if ( vi.isStart() )
            break;
        res.push_back( vi.back );
        v = topology_.dest( vi.back );
    }
    return res;
}

bool EdgePathsBuilder::addNextStep_( const VertPathInfo & c )
{
    auto & vi = vertPathInfoMap_[topology_.org( c.back )];
    if ( vi.metric > c.metric )
    {
        vi = c;
        nextSteps_.push( c );
        return true;
    }
    return false;
}

bool EdgePathsBuilder::addOrgRingSteps_( float orgMetric, EdgeId back, const VertBitSet * skipRegion )
{
    bool aNextStepAdded = false;
    for ( EdgeId e : orgRing( topology_, back ) )
    {
        if ( skipRegion && skipRegion->test( topology_.dest( e ) ) )
            continue;
        VertPathInfo c;
        c.back = e.sym();
        c.metric = orgMetric + metric_( e );
        aNextStepAdded = addNextStep_( c ) || aNextStepAdded;
    }
    return aNextStepAdded;
}

auto EdgePathsBuilder::growOneEdge() -> VertPathInfo
{
    while ( !nextSteps_.empty() )
    {
        const auto c = nextSteps_.top();
        nextSteps_.pop();
        auto & vi = vertPathInfoMap_[topology_.org( c.back )];
        if ( vi.metric < c.metric )
        {
            // shorter path to the vertex was found
            continue;
        }
        assert( vi.metric == c.metric );
        if ( keepGrowing_ )
            addOrgRingSteps_( c.metric, c.back );
        return c;
    }
    return {};
}

// internal version with pre-initialized builder
static EdgePath buildSmallestMetricPath( const MeshTopology& topology, VertId start, EdgePathsBuilder& b, float maxPathMetric )
{
    for ( ;;)
    {
        auto vinfo = b.growOneEdge();
        if ( !vinfo.back.valid() )
        {
            // unable to find the path
            return {};
        }
        if ( vinfo.metric > maxPathMetric )
        {
            // unable to find the path within given metric limitation
            return {};
        }
        if ( topology.org( vinfo.back ) == start )
            break;
    }
    return b.getPathBack( start );
}

EdgePath buildSmallestMetricPath( const MeshTopology& topology, const EdgeMetric& metric, VertId start, const VertBitSet& finish, float maxPathMetric /*= FLT_MAX */ )
{
    MR_TIMER

    EdgePathsBuilder b( topology, metric );
    b.addStartRegion( finish, 0 );
    return buildSmallestMetricPath( topology, start, b, maxPathMetric );
}

std::vector<EdgeId> buildSmallestMetricPath(
    const MeshTopology & topology, const EdgeMetric & metric,
    VertId start, VertId finish, float maxPathMetric )
{
    MR_TIMER

    EdgePathsBuilder b( topology, metric );
    b.addStart( topology.edgeWithOrg( finish ), 0 );
    return buildSmallestMetricPath( topology, start, b, maxPathMetric );
}

std::vector<EdgeId> buildSmallestMetricPathBiDir(
    const MeshTopology & topology, const EdgeMetric & metric,
    VertId start, VertId finish, float maxPathMetric )
{
    MR_TIMER

    if ( start == finish )
        return {};
    EdgePathsBuilder bs( topology, metric );
    bs.addStart( topology.edgeWithOrg( start ), 0 );
    EdgePathsBuilder bf( topology, metric );
    bf.addStart( topology.edgeWithOrg( finish ), 0 );

    VertId join;
    float joinPathMetric = maxPathMetric;
    for (;;)
    {
        auto ds = bs.doneDistance();
        auto df = bf.doneDistance();
        if ( bs.keepGrowing() && join && joinPathMetric <= ds + df )
        {
            bs.stopGrowing();
            bf.stopGrowing();
        }
        if ( ds <= df )
        {
            if ( ds >= FLT_MAX )
                break;
            auto c = bs.growOneEdge();
            auto v = topology.org( c.back );
            if ( auto info = bf.getVertInfo( v ) )
            {
                auto newMetric = c.metric + info->metric;
                if ( newMetric < joinPathMetric )
                {
                    join = v;
                    joinPathMetric = newMetric;
                }
            }
        }
        else
        {
            auto c = bf.growOneEdge();
            auto v = topology.org( c.back );
            if ( auto info = bs.getVertInfo( v ) )
            {
                auto newMetric = c.metric + info->metric;
                if ( newMetric < joinPathMetric )
                {
                    join = v;
                    joinPathMetric = newMetric;
                }
            }
        }
    }

    std::vector<EdgeId> res;
    if ( join )
    {
        res = bs.getPathBack( join );
        reverse( res );
        auto tail = bf.getPathBack( join );
        res.insert( res.end(), tail.begin(), tail.end() );
        assert( isEdgePath( topology, res ) );
        assert( topology.org( res.front() ) == start );
        assert( topology.dest( res.back() ) == finish );
    }
    return res;
}

std::vector<EdgeId> buildShortestPath( const Mesh & mesh, VertId start, VertId finish, float maxPathLen )
{
    return buildSmallestMetricPath( mesh.topology, edgeLengthMetric( mesh ), start, finish, maxPathLen );
}

std::vector<EdgeId> buildShortestPathBiDir( const Mesh & mesh, VertId start, VertId finish, float maxPathLen )
{
    return buildSmallestMetricPathBiDir( mesh.topology, edgeLengthMetric( mesh ), start, finish, maxPathLen );
}

EdgePath buildShortestPath( const Mesh& mesh, VertId start, const VertBitSet& finish, float maxPathLen /*= FLT_MAX */ )
{
    return buildSmallestMetricPath( mesh.topology, edgeLengthMetric( mesh ), start, finish, maxPathLen );
}

std::vector<VertId> getVertexOrdering( const MeshTopology & topology, VertBitSet region )
{
    MR_TIMER

    auto metric = [&]( EdgeId e ) 
    { 
        return region.test( topology.dest( e ) ) ? 1.0f : FLT_MAX; 
    }; 
    EdgePathsBuilder b( topology, metric );

    std::vector<VertId> res;
    res.reserve( region.count() );
    auto addToRes = [&]( VertId v )
    {
        region.reset( v );
        res.push_back( v );
    };

    while ( auto v0 = region.find_first() )
    {
        addToRes( v0 );
        b.addStart( topology.edgeWithOrg( v0 ), 0 );
        while ( b.doneDistance() < FLT_MAX )
        {
            auto vinfo = b.growOneEdge();
            addToRes( topology.org( vinfo.back ) );
        }
    }
    return res;
}

std::vector<EdgeLoop> extractClosedLoops( const MeshTopology & topology, EdgeBitSet & edges )
{
    MR_TIMER
    std::vector<EdgeLoop> res;
    for ( ;; )
    {
        UnionFind<VertId> vertComponents( topology.vertSize() );
        EdgeId loopEdge;
        for ( EdgeId e : edges )
        {
            const auto o = topology.org( e );
            const auto d = topology.dest( e );
            assert( o != d );
            if ( vertComponents.united( o, d ) )
            {
                loopEdge = e;
                break;
            }
            vertComponents.unite( o, d );
        }
        if ( !loopEdge.valid() )
            break;

        edges.reset( loopEdge );
        auto path = buildSmallestMetricPath( topology, [&]( EdgeId e )
        {
            return edges.test( e.sym() ) ? 1.0f : 1e5f; // sym() because inside buildSmallestMetricPath we search from finish to start
        }, topology.dest( loopEdge ), topology.org( loopEdge ) );
        assert ( !path.empty() );
        for ( EdgeId e : path )
        {
            assert( edges.test( e ) );
            edges.reset( e );
        }

        path.push_back( loopEdge );
        assert( isEdgeLoop( topology, path ) );

        res.push_back( std::move( path ) );
    }

    return res;
}

std::vector<EdgeLoop> extractClosedLoops( const MeshTopology & topology, const std::vector<EdgeId> & inEdges, EdgeBitSet * outNotLoopEdges )
{
    MR_TIMER
    EdgeBitSet edges;
    for ( auto e : inEdges )
    {
        if ( !edges.autoResizeTestSet( e.sym(), false ) )
            edges.autoResizeSet( e );
    }
    auto res = extractClosedLoops( topology, edges );
    if ( outNotLoopEdges )
        *outNotLoopEdges = std::move( edges );
    return res;
}

EdgeLoop extractLongestClosedLoop( const Mesh & mesh, const std::vector<EdgeId> & inEdges )
{
    MR_TIMER
    auto loops = extractClosedLoops( mesh.topology, inEdges );
    if ( loops.empty() )
        return {};
    sortPathsByLength( loops, mesh );
    return std::move( loops.back() );
}

bool dilateRegionByMetric( const MeshTopology & topology, const EdgeMetric & metric, FaceBitSet & region, float dilation, ProgressCallback callback )
{
    MR_TIMER
    auto vertRegion = getIncidentVerts( topology, region );
    if ( !dilateRegionByMetric( topology, metric, vertRegion, dilation, callback ) )
        return false;

    region = getInnerFaces( topology, vertRegion );
    return true;
}

bool dilateRegionByMetric( const MeshTopology & topology, const EdgeMetric & metric, VertBitSet & region, float dilation, ProgressCallback callback )
{
    MR_TIMER

    EdgePathsBuilder builder( topology, metric );
    builder.addStartRegion( region, 0 );

    for ( int i = 0; !builder.done() && builder.doneDistance() <= dilation; ++i )
    {
        if ( ( i % 1024 == 0 ) && callback && !callback( builder.doneDistance() / dilation ) )
            return false;

        auto vinfo = builder.growOneEdge();
        if ( vinfo.back.valid() )
            region.autoResizeSet( topology.org( vinfo.back ) );
    }

    if ( callback && !callback( 1.0f ) )
        return false;

    return true;
}

bool dilateRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback )
{
    MR_TIMER
    auto vertRegion = getIncidentVerts( topology, region );
    if ( !dilateRegionByMetric( topology, metric, vertRegion, dilation, callback ) )
        return false;

    region = getInnerEdges( topology, vertRegion );
    return true;
}

bool erodeRegionByMetric( const MeshTopology & topology, const EdgeMetric & metric, FaceBitSet & region, float dilation, ProgressCallback callback )
{
    MR_TIMER
    region = topology.getValidFaces() - region;
    if ( !dilateRegionByMetric( topology, metric, region, dilation, callback ) )
        return false;

    region = topology.getValidFaces() - region;
    return true;
}

bool erodeRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, VertBitSet& region, float dilation, ProgressCallback callback )
{
    MR_TIMER
    auto faceRegion = getInnerFaces( topology, region );
    if ( !erodeRegionByMetric( topology, metric, faceRegion, dilation, callback ) )
        return false;

    region = getIncidentVerts( topology, faceRegion );
    return true;
}

bool erodeRegionByMetric( const MeshTopology& topology, const EdgeMetric& metric, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback )
{
    MR_TIMER
    auto vertRegion = getIncidentVerts( topology, region );
    if ( !erodeRegionByMetric( topology, metric, vertRegion, dilation, callback ) )
        return false;

    region = getInnerEdges( topology, vertRegion );
    return true;
}

bool dilateRegion( const Mesh& mesh, FaceBitSet& region, float dilation, ProgressCallback callback )
{
    return dilateRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

bool dilateRegion( const Mesh& mesh, VertBitSet& region, float dilation, ProgressCallback callback )
{
    return dilateRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

bool dilateRegion( const Mesh& mesh, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback )
{
    return dilateRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

bool erodeRegion( const Mesh& mesh, FaceBitSet & region, float dilation, ProgressCallback callback )
{
    return erodeRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

bool erodeRegion( const Mesh& mesh, VertBitSet & region, float dilation, ProgressCallback callback )
{
    return erodeRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

bool erodeRegion( const Mesh& mesh, UndirectedEdgeBitSet& region, float dilation, ProgressCallback callback )
{
    return erodeRegionByMetric( mesh.topology, edgeLengthMetric( mesh ), region, dilation, callback );
}

int getPathPlaneIntersections( const Mesh & mesh, const EdgePath & path, const Plane3f & plane,
    std::vector<MeshEdgePoint> * outIntersections )
{
    MR_TIMER;
    int found = 0;
    for ( auto e : path )
    {
        auto o = plane.distance( mesh.orgPnt( e ) );
        auto d = plane.distance( mesh.destPnt( e ) );
        if ( ( o <= 0 && d > 0 ) || ( o >= 0 && d < 0 ) )
        {
            float a = -o / ( d - o );
            if ( outIntersections )
                outIntersections->emplace_back( e, a );
            ++found;
        }
    }
    return found;
}

int getPathEdgesInPlane( const Mesh & mesh, const EdgePath & path, const Plane3f & plane, float tolerance,
    std::vector<EdgeId> * outInPlaneEdges )
{
    MR_TIMER;
    int found = 0;
    for ( auto e : path )
    {
        auto o = plane.distance( mesh.orgPnt( e ) );
        auto d = plane.distance( mesh.destPnt( e ) );
        if ( std::abs( o ) <= tolerance && std::abs( d ) <= tolerance )
        {
            if ( outInPlaneEdges )
                outInPlaneEdges->emplace_back( e );
            ++found;
        }
    }
    return found;
}

TEST(MRMesh, BuildShortestPath) 
{
    Mesh cube = makeCube();
    auto path = buildShortestPath( cube, 0_v, 6_v );
    EXPECT_EQ( path.size(), 2 );
    EXPECT_EQ( cube.topology.org( path[0] ), 0_v );
    EXPECT_EQ( cube.topology.dest( path[0] ), cube.topology.org( path[1] ) );
    EXPECT_EQ( cube.topology.dest( path[1] ), 6_v );

    auto path34 = buildShortestPath( cube, 3_v, 4_v );
    EXPECT_EQ( path34.size(), 2 );

    std::vector<EdgePath> paths{ path, path34 };
    auto euclid = edgeLengthMetric( cube );
    EXPECT_GT( calcPathMetric( paths[0], euclid ), calcPathMetric( paths[1], euclid ) );
    sortPathsByMetric( paths, euclid );
    EXPECT_LE( calcPathMetric( paths[0], euclid ), calcPathMetric( paths[1], euclid ) );
}

} //namespace MR
