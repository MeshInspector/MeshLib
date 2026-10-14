// Crossover benchmark for MRRegionBoundary functions (MeshInspector/MeshLib#7134):
// for each function, times the parallel scan of the whole mesh (master) against the walk of the input set (the PR)
// on regions from 1/1024 to 1/4 of the mesh, in three layouts:
//   packed-contiguous   - mesh after packOptimally(), the first N faces
//   packed-scattered    - mesh after packOptimally(), N random faces
//   unpacked-contiguous - mesh in file order, the first N faces
// Both implementations are copies of the library code, so the program needs only a released MeshLib.
// Output: CSV lines  mesh,layout,function,fraction,setSize,scanMs,walkMs  ; a mismatch of the two results aborts.
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshTopology.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRBitSetParallelFor.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRBuffer.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace MR;

namespace scan // master: test every element of the mesh
{
UndirectedEdgeBitSet incEdgesF( const MeshTopology & t, const FaceBitSet & faces )
{
    UndirectedEdgeBitSet res( t.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        EdgeId e( ue );
        if ( contains( faces, t.left( e ) ) || contains( faces, t.right( e ) ) )
            res.set( ue );
    } );
    return res;
}
VertBitSet incVertsF( const MeshTopology & t, const FaceBitSet & faces )
{
    VertBitSet res = t.getValidVerts();
    BitSetParallelFor( res, [&]( VertId v )
    {
        for ( auto e : orgRing( t, v ) )
            if ( auto f = t.left( e ); f.valid() && faces.test( f ) )
                return;
        res.reset( v );
    } );
    return res;
}
VertBitSet innerVertsF( const MeshTopology & t, const FaceBitSet & region )
{
    VertBitSet res = t.getValidVerts();
    BitSetParallelFor( res, [&]( VertId v )
    {
        for ( auto e : orgRing( t, v ) )
        {
            auto f = t.left( e );
            if ( !f.valid() || !region.test( f ) )
            {
                res.reset( v );
                break;
            }
        }
    } );
    return res;
}
template <typename Test>
VertBitSet vertsOver( const MeshTopology & t, const VertBitSet & candidates, Test && test )
{
    VertBitSet res( t.vertSize() );
    BitSetParallelFor( candidates, [&]( VertId v ) { if ( test( v ) ) res.set( v ); } );
    return res;
}
bool isRegionBd( const MeshTopology & t, const FaceBitSet & region, VertId v )
{
    bool in = false, out = false;
    for ( auto e : orgRing( t, v ) )
    {
        auto l = t.left( e );
        if ( !l )
            continue;
        ( region.test( l ) ? in : out ) = true;
        if ( in && out )
            return true;
    }
    return false;
}
bool isRegionInner( const MeshTopology & t, const FaceBitSet & region, VertId v )
{
    bool in = false;
    for ( auto e : orgRing( t, v ) )
    {
        auto l = t.left( e );
        if ( !l )
            continue;
        if ( !region.test( l ) )
            return false;
        in = true;
    }
    return in;
}
VertBitSet regionBdVerts( const MeshTopology & t, const FaceBitSet & region )
    { return vertsOver( t, t.getValidVerts(), [&]( VertId v ) { return isRegionBd( t, region, v ); } ); }
VertBitSet regionInnerVerts( const MeshTopology & t, const FaceBitSet & region )
    { return vertsOver( t, t.getValidVerts(), [&]( VertId v ) { return isRegionInner( t, region, v ); } ); }
UndirectedEdgeBitSet innerEdgesF( const MeshTopology & t, const FaceBitSet & region )
{
    UndirectedEdgeBitSet res( t.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        EdgeId e( ue );
        if ( contains( region, t.left( e ) ) && contains( region, t.right( e ) ) )
            res.set( ue );
    } );
    return res;
}
UndirectedEdgeBitSet bdEdgesInside( const MeshTopology & t, const FaceBitSet & region )
{
    UndirectedEdgeBitSet res( t.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        const auto l = t.left( ue ), r = t.right( ue );
        if ( l && r && region.test( l ) != region.test( r ) )
            res.set( ue );
    } );
    return res;
}
EdgeBitSet leftBdEdges( const MeshTopology & t, const FaceBitSet & region )
{
    EdgeBitSet res( t.edgeSize() );
    BitSetParallelForAll( res, [&]( EdgeId e )
    {
        if ( !t.isLoneEdge( e ) && t.isLeftBdEdge( e, &region ) )
            res.set( e );
    } );
    return res;
}
FaceBitSet incFacesV( const MeshTopology & t, const VertBitSet & verts )
{
    FaceBitSet res( t.faceSize() );
    BitSetParallelFor( t.getValidFaces(), [&]( FaceId f )
    {
        for ( auto e : leftRing( t, f ) )
            if ( verts.test( t.org( e ) ) )
            {
                res.set( f );
                return;
            }
    } );
    return res;
}
FaceBitSet innerFacesOver( const MeshTopology & t, const VertBitSet & verts, const FaceBitSet & candidates )
{
    FaceBitSet res( t.faceSize() );
    BitSetParallelFor( candidates, [&]( FaceId f )
    {
        for ( auto e : leftRing( t, f ) )
            if ( !verts.test( t.org( e ) ) )
                return;
        res.set( f );
    } );
    return res;
}
FaceBitSet innerFacesV( const MeshTopology & t, const VertBitSet & verts ) { return innerFacesOver( t, verts, t.getValidFaces() ); }
UndirectedEdgeBitSet innerEdgesV( const MeshTopology & t, const VertBitSet & verts )
{
    UndirectedEdgeBitSet res( t.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        EdgeId e( ue );
        if ( contains( verts, t.org( e ) ) && contains( verts, t.dest( e ) ) )
            res.set( ue );
    } );
    return res;
}
UndirectedEdgeBitSet incEdgesE( const MeshTopology & t, const UndirectedEdgeBitSet & edges )
{
    UndirectedEdgeBitSet res = edges;
    res.resize( t.undirectedEdgeSize() );
    BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
    {
        if ( res.test( ue ) )
            return;
        EdgeId e( ue );
        for ( EdgeId ei : orgRing0( t, e ) )
            if ( edges.test( ei.undirected() ) )
            {
                res.set( ue );
                return;
            }
        for ( EdgeId ei : orgRing0( t, e.sym() ) )
            if ( edges.test( ei.undirected() ) )
            {
                res.set( ue );
                return;
            }
    } );
    return res;
}
} // namespace scan

namespace walk // the PR: walk only the input set and its neighbors
{
void addVertsOfFaces( const MeshTopology & t, const FaceBitSet & faces, VertBitSet & res )
{
    for ( auto f : faces )
        for ( auto e : leftRing( t, f ) )
            res.set( t.org( e ) );
}
VertBitSet vertsOfFaces( const MeshTopology & t, const FaceBitSet & faces, size_t size )
{
    VertBitSet res( size );
    addVertsOfFaces( t, faces, res );
    return res;
}
UndirectedEdgeBitSet incEdgesF( const MeshTopology & t, const FaceBitSet & faces )
{
    UndirectedEdgeBitSet res( t.undirectedEdgeSize() );
    for ( auto f : faces )
        for ( auto e : leftRing( t, f ) )
            res.set( e.undirected() );
    return res;
}
VertBitSet incVertsF( const MeshTopology & t, const FaceBitSet & faces ) { return vertsOfFaces( t, faces, t.getValidVerts().size() ); }
VertBitSet innerVertsF( const MeshTopology & t, const FaceBitSet & region )
{
    VertBitSet res = vertsOfFaces( t, region, t.getValidVerts().size() );
    BitSetParallelFor( res, [&]( VertId v )
    {
        for ( auto e : orgRing( t, v ) )
        {
            auto f = t.left( e );
            if ( !f.valid() || !region.test( f ) )
            {
                res.reset( v );
                break;
            }
        }
    } );
    return res;
}
VertBitSet regionBdVerts( const MeshTopology & t, const FaceBitSet & region )
    { return scan::vertsOver( t, vertsOfFaces( t, region, t.vertSize() ), [&]( VertId v ) { return scan::isRegionBd( t, region, v ); } ); }
VertBitSet regionInnerVerts( const MeshTopology & t, const FaceBitSet & region )
    { return scan::vertsOver( t, vertsOfFaces( t, region, t.vertSize() ), [&]( VertId v ) { return scan::isRegionInner( t, region, v ); } ); }
UndirectedEdgeBitSet innerEdgesF( const MeshTopology & t, const FaceBitSet & region )
{
    UndirectedEdgeBitSet res( t.undirectedEdgeSize() );
    for ( auto f : region )
        for ( auto e : leftRing( t, f ) )
            if ( contains( region, t.right( e ) ) )
                res.set( e.undirected() );
    return res;
}
UndirectedEdgeBitSet bdEdgesInside( const MeshTopology & t, const FaceBitSet & region )
{
    UndirectedEdgeBitSet res( t.undirectedEdgeSize() );
    for ( auto f : region )
        for ( auto e : leftRing( t, f ) )
            if ( auto r = t.right( e ); r && !region.test( r ) )
                res.set( e.undirected() );
    return res;
}
EdgeBitSet leftBdEdges( const MeshTopology & t, const FaceBitSet & region )
{
    EdgeBitSet res( t.edgeSize() );
    for ( auto f : region )
        for ( auto e : leftRing( t, f ) )
            if ( t.isLeftBdEdge( e, &region ) )
                res.set( e );
    return res;
}
FaceBitSet incFacesV( const MeshTopology & t, const VertBitSet & verts )
{
    FaceBitSet res( t.faceSize() );
    for ( auto v : verts )
        for ( auto e : orgRing( t, v ) )
            if ( auto l = t.left( e ) )
                res.set( l );
    return res;
}
FaceBitSet innerFacesV( const MeshTopology & t, const VertBitSet & verts ) { return scan::innerFacesOver( t, verts, incFacesV( t, verts ) ); }
UndirectedEdgeBitSet innerEdgesV( const MeshTopology & t, const VertBitSet & verts )
{
    UndirectedEdgeBitSet res( t.undirectedEdgeSize() );
    for ( auto v : verts )
        for ( auto e : orgRing( t, v ) )
            if ( verts.test( t.dest( e ) ) )
                res.set( e.undirected() );
    return res;
}
UndirectedEdgeBitSet incEdgesE( const MeshTopology & t, const UndirectedEdgeBitSet & edges )
{
    UndirectedEdgeBitSet res = edges;
    res.resize( t.undirectedEdgeSize() );
    for ( auto ue : edges )
    {
        if ( ue >= t.undirectedEdgeSize() )
            break;
        for ( EdgeId e : { EdgeId( ue ), EdgeId( ue ).sym() } )
            for ( EdgeId ei : orgRing( t, e ) )
                res.set( ei.undirected() );
    }
    return res;
}
} // namespace walk

static double msSince( std::chrono::steady_clock::time_point t0 )
{
    return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
}

int main( int argc, char ** argv )
{
    if ( argc < 3 )
    {
        std::fprintf( stderr, "usage: region_crossovers <reps> <mesh> [<mesh>...]\n" );
        return 1;
    }
    const int reps = std::atoi( argv[1] );
    std::fprintf( stderr, "hardware threads: %u\n", std::thread::hardware_concurrency() );
    std::printf( "mesh,layout,function,fraction,setSize,scanMs,walkMs\n" );
    const int fractions[] = { 1024, 512, 256, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4 };

    for ( int mi = 2; mi < argc; ++mi )
    {
        const std::string meshName = std::string( argv[mi] ).substr( std::string( argv[mi] ).find_last_of( "/\\" ) + 1 );
        auto loaded = MeshLoad::fromAnySupportedFormat( argv[mi] );
        if ( !loaded )
        {
            std::fprintf( stderr, "%s: %s\n", argv[mi], loaded.error().c_str() );
            return 1;
        }
        for ( const char * layout : { "packed-contiguous", "packed-scattered", "unpacked-contiguous" } )
        {
            Mesh mesh = *loaded;
            if ( std::string( layout ).starts_with( "packed" ) )
                mesh.packOptimally();
            const auto & t = mesh.topology;
            std::vector<FaceId> order;
            for ( auto f : t.getValidFaces() )
                order.push_back( f );
            if ( std::string( layout ).ends_with( "scattered" ) )
                std::shuffle( order.begin(), order.end(), std::mt19937( 42 ) );

            for ( int fr : fractions )
            {
                FaceBitSet region( t.faceSize() );
                for ( size_t i = 0; i < order.size() / fr; ++i )
                    region.set( order[i] );
                const auto verts = scan::incVertsF( t, region );
                const auto edges = scan::incEdgesF( t, region );

                auto run = [&]( const char * name, size_t setSize, auto && fScan, auto && fWalk )
                {
                    if ( !( fScan() == fWalk() ) )
                    {
                        std::fprintf( stderr, "MISMATCH %s %s %s 1/%d\n", meshName.c_str(), layout, name, fr );
                        std::exit( 2 );
                    }
                    double bs = 1e30, bw = 1e30;
                    for ( int r = 0; r < reps; ++r )
                    {
                        auto t0 = std::chrono::steady_clock::now();
                        { [[maybe_unused]] auto x = fScan(); }
                        bs = std::min( bs, msSince( t0 ) );
                        t0 = std::chrono::steady_clock::now();
                        { [[maybe_unused]] auto x = fWalk(); }
                        bw = std::min( bw, msSince( t0 ) );
                    }
                    std::printf( "%s,%s,%s,%d,%zu,%.4f,%.4f\n", meshName.c_str(), layout, name, fr, setSize, bs, bw );
                    std::fflush( stdout );
                };
                const size_t nf = region.count(), nv = verts.count(), ne = edges.count();
                run( "getIncidentEdges(faces)", nf, [&] { return scan::incEdgesF( t, region ); }, [&] { return walk::incEdgesF( t, region ); } );
                run( "getIncidentVerts(faces)", nf, [&] { return scan::incVertsF( t, region ); }, [&] { return walk::incVertsF( t, region ); } );
                run( "getInnerVerts(faces)", nf, [&] { return scan::innerVertsF( t, region ); }, [&] { return walk::innerVertsF( t, region ); } );
                run( "getRegionBoundaryVerts", nf, [&] { return scan::regionBdVerts( t, region ); }, [&] { return walk::regionBdVerts( t, region ); } );
                run( "getRegionInnerVerts", nf, [&] { return scan::regionInnerVerts( t, region ); }, [&] { return walk::regionInnerVerts( t, region ); } );
                run( "getInnerEdges(faces)", nf, [&] { return scan::innerEdgesF( t, region ); }, [&] { return walk::innerEdgesF( t, region ); } );
                run( "findRegionBoundaryUndirectedEdgesInsideMesh", nf, [&] { return scan::bdEdgesInside( t, region ); }, [&] { return walk::bdEdgesInside( t, region ); } );
                run( "findAllLeftBdEdges", nf, [&] { return scan::leftBdEdges( t, region ); }, [&] { return walk::leftBdEdges( t, region ); } );
                run( "getIncidentFaces(verts)", nv, [&] { return scan::incFacesV( t, verts ); }, [&] { return walk::incFacesV( t, verts ); } );
                run( "getInnerFaces(verts)", nv, [&] { return scan::innerFacesV( t, verts ); }, [&] { return walk::innerFacesV( t, verts ); } );
                run( "getInnerEdges(verts)", nv, [&] { return scan::innerEdgesV( t, verts ); }, [&] { return walk::innerEdgesV( t, verts ); } );
                run( "getIncidentEdges(edges)", ne, [&] { return scan::incEdgesE( t, edges ); }, [&] { return walk::incEdgesE( t, edges ); } );
            }
        }
    }
    return 0;
}
