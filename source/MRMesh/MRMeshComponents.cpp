#include "MRMeshComponents.h"
#include "MRMesh.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRRegionBoundary.h"
#include "MRMeshBuilder.h"
#include "MREdgeIterator.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"
#include <parallel_hashmap/phmap.h>

namespace MR
{
namespace MeshComponents
{

std::pair<std::vector<int>, int> getUniqueRoots( const FaceMap& allRoots, const FaceBitSet& region )
{
    constexpr int InvalidRoot = -1;
    std::vector<int> uniqueRootsMap( allRoots.size(), InvalidRoot );
    int k = 0;
    int curRoot;
    for ( auto f : region )
    {
        curRoot = allRoots[f];
        auto& uniqIndex = uniqueRootsMap[curRoot];
        if ( uniqIndex == InvalidRoot )
        {
            uniqIndex = k;
            ++k;
        }
    }
    return { std::move( uniqueRootsMap ),k };
}

FaceBitSet getComponent( const MeshPart& meshPart, FaceId id, FaceIncidence incidence/* = FaceIncidence::PerEdge*/ )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructureFaces( meshPart, incidence );
    const FaceBitSet& region = meshPart.mesh.topology.getFaceIds( meshPart.region );

    int faceRoot = unionFindStruct.find( id );
    const auto& allRoots = unionFindStruct.roots();
    FaceBitSet res;
    res.resize( allRoots.size() );
    for ( auto f : region )
    {
        if ( allRoots[f] == faceRoot )
            res.set( f );
    }
    return res;
}

VertBitSet getComponentVerts( const Mesh& mesh, VertId id, const VertBitSet* region /*= nullptr */ )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructureVerts( mesh, region );
    const VertBitSet& vertsRegion = mesh.topology.getVertIds( region );

    int vertRoot = unionFindStruct.find( id );
    const auto& allRoots = unionFindStruct.roots();
    VertBitSet res;
    res.resize( allRoots.size() );
    for ( auto v : vertsRegion )
    {
        if ( allRoots[v] == vertRoot )
            res.set( v );
    }
    return res;

}

FaceBitSet getLargestComponent( const MeshPart& meshPart, FaceIncidence incidence /*= FaceIncidence::PerEdge */ )
{
    MR_TIMER;

    auto unionFindStruct = getUnionFindStructureFaces( meshPart, incidence );
    const auto& mesh = meshPart.mesh;
    const FaceBitSet& region = mesh.topology.getFaceIds( meshPart.region );

    const auto& allRoots = unionFindStruct.roots();
    auto [uniqueRootsMap, k] = getUniqueRoots( allRoots, region );

    double maxArea = -DBL_MAX;
    int maxI = 0;
    std::vector<double> areas( k, 0.0 );
    for ( auto f : region )
    {
        auto index = uniqueRootsMap[allRoots[f]];
        auto& area = areas[index];
        area += meshPart.mesh.dblArea( f );
        if ( area > maxArea )
        {
            maxI = index;
            maxArea = area;
        }
    }
    FaceBitSet maxAreaComponent( region.find_last() + 1 );
    for ( auto f : region )
    {
        auto index = uniqueRootsMap[allRoots[f]];
        if ( index != maxI )
            continue;
        maxAreaComponent.set( f );
    }
    return maxAreaComponent;
}

VertBitSet getLargestComponentVerts( const Mesh& mesh, const VertBitSet* region /*= nullptr */ )
{
    MR_TIMER;

    auto allComponents = getAllComponentsVerts( mesh, region );

    if ( allComponents.empty() )
        return {};

    return *std::max_element( allComponents.begin(), allComponents.end(), []( const VertBitSet& a, const VertBitSet& b )
    {
        return a.count() < b.count();
    } );
}

FaceBitSet getComponents( const MeshPart& meshPart, const FaceBitSet & seeds, FaceIncidence incidence/* = FaceIncidence::PerEdge*/ )
{
    MR_TIMER;

    FaceBitSet res;
    if ( seeds.empty() )
        return res;

    auto unionFindStruct = getUnionFindStructureFaces( meshPart, incidence );
    const FaceBitSet& region = meshPart.mesh.topology.getFaceIds( meshPart.region );

    FaceId faceRoot;
    for ( auto s : seeds )
    {
        if ( faceRoot < 0 )
            faceRoot = unionFindStruct.find( s );
        else
            faceRoot = unionFindStruct.unite( faceRoot, s );
    }

    const auto& allRoots = unionFindStruct.roots();
    res.resize( allRoots.size() );
    for ( auto f : region )
    {
        if ( allRoots[f] == faceRoot )
            res.set( f );
    }
    return res;
}

VertBitSet getComponentsVerts( const Mesh& mesh, const VertBitSet& seeds, const VertBitSet* region /*= nullptr */ )
{
    MR_TIMER;

    VertBitSet res;
    if ( seeds.empty() )
        return res;

    auto unionFindStruct = getUnionFindStructureVerts( mesh, region );
    const VertBitSet& vertRegion = mesh.topology.getVertIds( region );

    VertId vertRoot;
    for ( auto s : seeds )
    {
        if ( vertRoot < 0 )
            vertRoot = unionFindStruct.find( s );
        else
            vertRoot = unionFindStruct.unite( vertRoot, s );
    }

    const auto& allRoots = unionFindStruct.roots();
    res.resize( allRoots.size() );
    for ( auto v : vertRegion )
    {
        if ( allRoots[v] == vertRoot )
            res.set( v );
    }
    return res;

}

size_t getNumComponents( const MeshPart& meshPart, FaceIncidence incidence )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructureFaces( meshPart, incidence );
    const auto& mesh = meshPart.mesh;
    const FaceBitSet& region = mesh.topology.getFaceIds( meshPart.region );

    const auto& allRoots = unionFindStruct.roots();
    FaceHashSet componentRoots;
    for ( auto f : region )
    {
        componentRoots.insert( allRoots[f] );
    }
    return componentRoots.size();
}

std::vector<FaceBitSet> getAllComponents( const MeshPart& meshPart, FaceIncidence incidence/* = FaceIncidence::PerEdge*/ )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructureFaces( meshPart, incidence );
    const auto& mesh = meshPart.mesh;
    const FaceBitSet& region = mesh.topology.getFaceIds( meshPart.region );

    const auto& allRoots = unionFindStruct.roots();
    auto [uniqueRootsMap, k] = getUniqueRoots( allRoots, region );
    std::vector<FaceBitSet> res( k );
    // this block is needed to limit allocations for not packed meshes
    std::vector<int> resSizes( k, 0 );
    for ( auto f : region )
    {
        int index = uniqueRootsMap[allRoots[f]];
        if ( f > resSizes[index] )
            resSizes[index] = f;
    }
    for ( int i = 0; i < k; ++i )
        res[i].resize( resSizes[i] + 1 );
    // end of allocation block
    for ( auto f : region )
    {
        res[uniqueRootsMap[allRoots[f]]].set( f );
    }
    return res;
}

static std::vector<VertBitSet> getAllComponentsVerts( UnionFind<VertId>& unionFindStruct, const VertBitSet& vertsRegion, const VertBitSet* doNotOutput )
{
    MR_TIMER;

    const auto& allRoots = unionFindStruct.roots();
    constexpr int InvalidRoot = -1;
    std::vector<int> uniqueRootsMap( allRoots.size(), InvalidRoot );
    int k = 0;
    int curRoot;
    for ( auto v : vertsRegion )
    {
        if ( doNotOutput && doNotOutput->test( v ) )
            continue;
        curRoot = allRoots[v];
        auto& uniqIndex = uniqueRootsMap[curRoot];
        if ( uniqIndex == InvalidRoot )
        {
            uniqIndex = k;
            ++k;
        }
    }
    std::vector<VertBitSet> res( k, VertBitSet( allRoots.size() ) );
    for ( auto v : vertsRegion )
    {
        if ( doNotOutput && doNotOutput->test( v ) )
            continue;
        curRoot = allRoots[v];
        res[uniqueRootsMap[curRoot]].set( v );
    }
    return res;
}

std::vector<VertBitSet> getAllComponentsVerts( const Mesh& mesh, const VertBitSet* region )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructureVerts( mesh, region );
    const VertBitSet& vertsRegion = mesh.topology.getVertIds( region );
    return getAllComponentsVerts( unionFindStruct, vertsRegion, nullptr );
}

std::vector<VertBitSet> getAllComponentsVertsSeparatedByPath( const Mesh& mesh, const SurfacePath& path )
{
    VertBitSet pathVerts;
    auto unionFindStruct = getUnionFindStructureVertsSeparatedByPath( mesh, path, &pathVerts );
    const VertBitSet& vertsRegion = mesh.topology.getValidVerts();
    return getAllComponentsVerts( unionFindStruct, vertsRegion, &pathVerts );
}

std::vector<EdgeBitSet> getAllComponentsEdges( const Mesh& mesh, const EdgeBitSet & edges )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructureVerts( mesh, edges );

    const auto& allRoots = unionFindStruct.roots();
    constexpr int InvalidRoot = -1;
    std::vector<int> uniqueRootsMap( allRoots.size(), InvalidRoot );
    int k = 0;
    EdgeId eMax;
    for ( auto e : edges )
    {
        if ( eMax < e )
            eMax = e;
        int curRoot = allRoots[ mesh.topology.org( e ) ];
        auto& uniqIndex = uniqueRootsMap[curRoot];
        if ( uniqIndex == InvalidRoot )
        {
            uniqIndex = k;
            ++k;
        }
    }
    std::vector<EdgeBitSet> res( k, EdgeBitSet( eMax + 1 ) );
    for ( auto e : edges )
    {
        int curRoot = allRoots[ mesh.topology.org( e ) ];
        res[uniqueRootsMap[curRoot]].set( e );
    }
    return res;
}

UnionFind<FaceId> getUnionFindStructureFaces( const MeshPart& meshPart, FaceIncidence incidence/* = FaceIncidence::PerEdge*/ )
{
    MR_TIMER;

    const auto& mesh = meshPart.mesh;
    const FaceBitSet& region = mesh.topology.getFaceIds( meshPart.region );
    UnionFind<FaceId> unionFindStructure( region.find_last() + 1 );
    for ( auto f0 : region )
    { 
        if ( incidence == FaceIncidence::PerEdge )
        {
            EdgeId e[3];
            mesh.topology.getTriEdges( f0, e );
            for ( int i = 0; i < 3; ++i )
            {
                assert( mesh.topology.left( e[i] ) == f0 );
                FaceId f1 = mesh.topology.right( e[i] );
                if ( f1 < f0 && contains( meshPart.region, f1 ) )
                    unionFindStructure.unite( f0, f1 );
            }
        }
        else if ( incidence == FaceIncidence::PerVertex )
        {
            VertId vid[3];
            mesh.topology.getTriVerts( f0, vid );
            for ( auto faceVert : vid )
            {
                for ( auto edge : orgRing( mesh.topology, faceVert ) )
                {
                    FaceId f1 = mesh.topology.left( edge );
                    if ( f1 < f0 && contains( meshPart.region, f1 ) )
                        unionFindStructure.unite( f0, f1 );
                }
            }
        }
    }
    return unionFindStructure;
}

UnionFind<VertId> getUnionFindStructureVerts( const Mesh& mesh, const VertBitSet* region )
{
    MR_TIMER;

    const VertBitSet& vertsRegion = mesh.topology.getVertIds( region );

    auto test = [region]( VertId v )
    {
        if ( !region )
            return true;
        else
            return region->test( v );
    };

    UnionFind<VertId> unionFindStructure( mesh.topology.lastValidVert() + 1 );

    VertId v1;
    for ( auto v0 : vertsRegion )
    {
        for ( auto e : orgRing( mesh.topology, v0 ) )
        {
            v1 = mesh.topology.dest( e );
            if ( v1.valid() && test( v1 ) && v1 < v0 )
                unionFindStructure.unite( v0, v1 );
        }
    }
    return unionFindStructure;
}

UnionFind<VertId> getUnionFindStructureVerts( const Mesh& mesh, const EdgeBitSet & edges )
{
    MR_TIMER;

    UnionFind<VertId> unionFindStructure( mesh.topology.lastValidVert() + 1 );

    for ( EdgeId e : edges )
    {
        auto vo = mesh.topology.org( e );
        auto vd = mesh.topology.dest( e );
        unionFindStructure.unite( vo, vd );
    }
    return unionFindStructure;
}

UnionFind<VertId> getUnionFindStructureVertsEx( const Mesh& mesh, const UndirectedEdgeBitSet & ignoreEdges )
{
    MR_TIMER;

    UnionFind<VertId> unionFindStructure( mesh.topology.lastValidVert() + 1 );

    for ( auto ue : undirectedEdges( mesh.topology ) )
    {
        if ( ignoreEdges.test( ue ) )
            continue;
        auto vo = mesh.topology.org( ue );
        auto vd = mesh.topology.dest( ue );
        unionFindStructure.unite( vo, vd );
    }
    return unionFindStructure;
}

UnionFind<VertId> getUnionFindStructureVertsSeparatedByPath( const Mesh& mesh, const SurfacePath& path, VertBitSet * outPathVerts )
{
    MR_TIMER;
    UndirectedEdgeBitSet ignoreEdges( mesh.topology.undirectedEdgeSize() );

    for ( const MeshEdgePoint & ep : path )
    {
        if ( VertId v = ep.inVertex( mesh.topology ) )
        {
            if ( outPathVerts )
                outPathVerts->autoResizeSet( v );
            for ( auto e : orgRing( mesh.topology, v ) )
                ignoreEdges.set( e.undirected() );
            continue;
        }
        ignoreEdges.set( ep.e.undirected() );
    }
    return getUnionFindStructureVertsEx( mesh, ignoreEdges );
}

TEST(MRMesh, getAllComponentsEdges) 
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );
    mesh.points.emplace_back( 0.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 1.f, 0.f );
    mesh.points.emplace_back( 0.f, 1.f, 0.f );

    EdgeId e12 = mesh.topology.findEdge( 1_v, 2_v );
    EdgeId e30 = mesh.topology.findEdge( 3_v, 0_v );

    EdgeBitSet ebs( 10 );
    ebs.set( e12 );
    ebs.set( e30 );
    auto comp = getAllComponentsEdges( mesh, ebs );
    ASSERT_EQ( comp.size(), 2 );
    ASSERT_EQ( comp[0].count(), 1 );
    ASSERT_EQ( comp[1].count(), 1 );

    ebs.set( e12.sym() );
    ebs.set( e30.sym() );
    comp = getAllComponentsEdges( mesh, ebs );
    ASSERT_EQ( comp.size(), 2 );
    ASSERT_EQ( comp[0].count(), 2 );
    ASSERT_EQ( comp[1].count(), 2 );

    ebs.set( mesh.topology.findEdge( 0_v, 1_v ) );
    comp = getAllComponentsEdges( mesh, ebs );
    ASSERT_EQ( comp.size(), 1 );
    ASSERT_EQ( comp[0].count(), 5 );
}

} // namespace MeshComponents

} // namespace MR
