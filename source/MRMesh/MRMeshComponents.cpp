#include "MRMeshComponents.h"
#include "MRMesh.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRRegionBoundary.h"
#include "MREdgeIterator.h"
#include "MRUnionFindParallel.h"
#include <parallel_hashmap/phmap.h>
#include <climits>

namespace MR
{
namespace MeshComponents
{

/// returns
/// 1. the mapping: Root Id -> Region Id in [0, 1, 2, ...)
/// 2. the total number of roots/regions
template<typename T>
static std::pair<Vector<RegionId, Id<T>>, int> getUniqueRootIds( const Vector<Id<T>, Id<T>>& allRoots, const TaggedBitSet<T>& region )
{
    MR_TIMER;
    Vector<RegionId, Id<T>> uniqueRootsMap( allRoots.size() );
    int k = 0;
    for ( auto f : region )
    {
        auto& uniqIndex = uniqueRootsMap[allRoots[f]];
        if ( uniqIndex < 0 )
        {
            uniqIndex = RegionId( k );
            ++k;
        }
        uniqueRootsMap[f] = uniqIndex;
    }
    return { std::move( uniqueRootsMap ), k };
}

FaceBitSet getComponent( const MeshPart& meshPart, FaceId id, FaceIncidence incidence, const UndirectedEdgeBitSet * isCompBd )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructureFaces( meshPart, incidence, isCompBd );
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

FaceBitSet getLargestComponent( const MeshPart& meshPart, FaceIncidence incidence, const UndirectedEdgeBitSet * isCompBd, float minArea, int * numSmallerComponents )
{
    MR_TIMER;

    auto unionFindStruct = getUnionFindStructureFaces( meshPart, incidence, isCompBd );
    const auto& mesh = meshPart.mesh;
    const FaceBitSet& region = mesh.topology.getFaceIds( meshPart.region );

    FaceBitSet maxAreaComponent;
    const auto& allRoots = unionFindStruct.roots();
    auto [uniqueRootsMap, k] = getUniqueRootIds( allRoots, region );
    if ( k <= 0 )
    {
        if ( numSmallerComponents )
            *numSmallerComponents = 0;
        return maxAreaComponent;
    }

    double maxDblArea = -DBL_MAX;
    int maxI = 0;
    std::vector<double> dblAreas( k, 0.0 );
    for ( auto f : region )
    {
        auto index = uniqueRootsMap[f];
        auto& dblArea = dblAreas[index];
        dblArea += meshPart.mesh.dblArea( f );
        if ( dblArea > maxDblArea )
        {
            maxI = index;
            maxDblArea = dblArea;
        }
    }
    if ( maxDblArea < 2 * minArea )
    {
        if ( numSmallerComponents )
            *numSmallerComponents = k;
        return maxAreaComponent;
    }
    if ( numSmallerComponents )
        *numSmallerComponents = k - 1;
    maxAreaComponent.resize( region.find_last() + 1 );
    for ( auto f : region )
    {
        auto index = uniqueRootsMap[f];
        if ( index != maxI )
            continue;
        maxAreaComponent.set( f );
    }
    return maxAreaComponent;
}

VertBitSet getLargestComponentVerts( const Mesh& mesh, const VertBitSet* region /*= nullptr */ )
{
    MR_TIMER;

    auto unionFindStruct = getUnionFindStructureVerts( mesh, region );

    VertId largestRoot;
    int largestNumVerts = 0;
    for ( auto r : findRootsBitSet( unionFindStruct, region ) )
    {
        if ( !largestRoot || largestNumVerts < unionFindStruct.sizeOfComp( r ) )
        {
            largestRoot = r;
            largestNumVerts = unionFindStruct.sizeOfComp( r );
        }
    }
    if ( !largestRoot )
        return {}; // e.g. empty region

    return findComponentBitSet( unionFindStruct, largestRoot, region );
}

VertBitSet getLargeComponentVerts( const Mesh& mesh, int minVerts, const VertBitSet* region )
{
    MR_TIMER;
    assert( minVerts >= 2 );
    if ( minVerts <= 1 )
        return mesh.topology.getVertIds( region );

    auto unionFind = getUnionFindStructureVerts( mesh, region );

    VertBitSet res( mesh.topology.vertSize() );
    for ( auto f : mesh.topology.getVertIds( region ) )
    {
        if ( unionFind.sizeOfComp( f ) >= minVerts )
            res.set( f );
    }
    return res;
}

FaceBitSet getComponents( const MeshPart& meshPart, const FaceBitSet & seeds, FaceIncidence incidence, const UndirectedEdgeBitSet * isCompBd )
{
    MR_TIMER;

    FaceBitSet res;
    if ( seeds.none() )
        return res;

    auto unionFindStruct = getUnionFindStructureFaces( meshPart, incidence, isCompBd );
    const FaceBitSet& region = meshPart.mesh.topology.getFaceIds( meshPart.region );

    FaceId faceRoot;
    for ( auto s : seeds )
    {
        if ( faceRoot < 0 )
            faceRoot = unionFindStruct.find( s );
        else
            faceRoot = unionFindStruct.unite( faceRoot, s ).first;
    }

    if ( faceRoot )
    {
        const auto& allRoots = unionFindStruct.roots();
        res.resize( allRoots.size() );
        BitSetParallelFor( region, [&]( FaceId f )
        {
            if ( allRoots[f] == faceRoot )
                res.set( f );
        } );
    }
    return res;
}

FaceBitSet getLargeByAreaComponents( const MeshPart& mp, float minArea, const UndirectedEdgeBitSet * isCompBd )
{
    auto unionFind = getUnionFindStructureFacesPerEdge( mp, isCompBd );
    return getLargeByAreaComponents( mp, unionFind, minArea );
}

Expected<FaceBitSet> expandToComponents( const MeshPart& mp, const FaceBitSet& seeds, const ExpandToComponentsParams& params /*= {} */ )
{
    if ( params.coverRatio > 1.0f )
        return FaceBitSet();
    if ( params.coverRatio <= 0.0f )
        return getComponents( mp, seeds, params.incidence, params.isCompBd );

    MR_TIMER;

    auto res = seeds;
    auto compMapRes = MeshComponents::getAllComponentsMap( mp, params.incidence, params.isCompBd );
    const auto& compMap = compMapRes.first;
    int numComps = compMapRes.second;

    if ( !reportProgress( params.cb, 0.3f ) )
        return unexpectedOperationCanceled();

    RegionBitSet compsWithSeeds( numComps );
    for ( auto f : res )
        compsWithSeeds.set( compMap[f] );

    if ( !reportProgress( params.cb, 0.6f ) )
        return unexpectedOperationCanceled();

    const auto& region = mp.mesh.topology.getFaceIds( mp.region );

    struct AreaCounter
    {
        float seedArea = 0.0f;
        float totalArea = 0.0f;
    };
    Vector<AreaCounter, RegionId> areas( numComps );
    for ( auto f : region )
    {
        auto rId = compMap[f];
        if ( !compsWithSeeds.test( rId ) )
            continue;
        auto area = mp.mesh.area( f );
        areas[rId].totalArea += area;
        if ( res.test( f ) )
            areas[rId].seedArea += area;
    }

    if ( !reportProgress( params.cb, 0.9f ) )
        return unexpectedOperationCanceled();

    auto largeSeedsCompsBs = compsWithSeeds;
    for ( auto rId : compsWithSeeds )
    {
        if ( areas[rId].seedArea / areas[rId].totalArea < params.coverRatio )
            largeSeedsCompsBs.reset( rId );
    }
    res.resize( region.size() );
    BitSetParallelFor( region, [&] ( FaceId f )
    {
        res.set( f, largeSeedsCompsBs.test( compMap[f] ) );
    } );
    if ( !reportProgress( params.cb, 1.0f ) )
        return unexpectedOperationCanceled();
    if ( params.optOutNumComponents )
        *params.optOutNumComponents = int( largeSeedsCompsBs.count() );
    return res;
}

FaceBitSet getLargeByAreaSmoothComponents( const MeshPart& mp, float minArea, float angleFromPlanar,
    UndirectedEdgeBitSet * outBdEdgesBetweenLargeComps )
{
    const float critCos = std::cos( angleFromPlanar );
    UndirectedEdgeBitSet bdEdges( mp.mesh.topology.undirectedEdgeSize() );
    BitSetParallelForAll( bdEdges, [&]( UndirectedEdgeId ue )
    {
        if ( mp.mesh.topology.isLoneEdge( ue ) )
            return;
        if ( mp.mesh.dihedralAngleCos( ue ) < critCos )
            bdEdges.set( ue );
    } );
    auto unionFind = MeshComponents::getUnionFindStructureFacesPerEdge( mp, &bdEdges );
    return MeshComponents::getLargeByAreaComponents( mp, unionFind, minArea, outBdEdgesBetweenLargeComps );
}

FaceBitSet getLargeByAreaComponents( const MeshPart& mp, UnionFind<FaceId> & unionFind, float minArea,
    UndirectedEdgeBitSet * outBdEdgesBetweenLargeComps )
{
    MR_TIMER;

    HashMap<FaceId, float> root2area;
    const FaceBitSet& region = mp.mesh.topology.getFaceIds( mp.region );
    for ( auto f : region )
    {
        auto root = unionFind.find( f );
        root2area[ root ] += mp.mesh.area( f );
    }

    FaceBitSet res( mp.mesh.topology.faceSize() );
    for ( auto f : region )
    {
        auto root = unionFind.find( f );
        if ( root2area[ root ] >= minArea )
            res.set( f );
    }

    if ( outBdEdgesBetweenLargeComps )
    {
        outBdEdgesBetweenLargeComps->clear();
        outBdEdgesBetweenLargeComps->resize( mp.mesh.topology.undirectedEdgeSize() );
        const auto & roots = unionFind.parents();
        BitSetParallelForAll( *outBdEdgesBetweenLargeComps, [&]( UndirectedEdgeId ue )
        {
            auto l = mp.mesh.topology.left( ue );
            if ( !l )
                return;
            auto lroot = roots[l];
            if ( root2area[ lroot ] < minArea )
                return;
            auto r = mp.mesh.topology.right( ue );
            if ( !r )
                return;
            auto rroot = roots[r];
            if ( root2area[ rroot ] < minArea )
                return;
            if ( lroot != rroot )
                outBdEdgesBetweenLargeComps->set( ue );
        } );
    }

    return res;
}

std::vector<FaceBitSet> getNLargeByAreaComponents( const MeshPart& mp, const LargeByAreaComponentsSettings & settings )
{
    MR_TIMER;
    std::vector<FaceBitSet> res;

    assert( settings.maxLargeComponents > 0 );
    if ( settings.maxLargeComponents <= 0 )
    {
        if ( settings.numSmallerComponents )
            *settings.numSmallerComponents = -1; //unknown
        return res;
    }
    if ( settings.maxLargeComponents == 1 )
    {
        res.push_back( getLargestComponent( mp, PerEdge, settings.isCompBd, settings.minArea, settings.numSmallerComponents ) );
        return res;
    }

    auto unionFind = getUnionFindStructureFacesPerEdge( mp, settings.isCompBd );
    const auto & roots = unionFind.roots();

    HashMap<FaceId, float> root2area;
    const FaceBitSet& region = mp.mesh.topology.getFaceIds( mp.region );
    for ( auto f : region )
        root2area[ roots[f] ] += mp.mesh.area( f );

    struct AreaRoot
    {
        float area = 0;
        FaceId root;
        constexpr auto operator <=>( const AreaRoot& ) const = default;
    };

    std::vector<AreaRoot> areaRootVec;
    areaRootVec.reserve( root2area.size() );
    // fill it with not too small components
    for ( const auto & [root, area] : root2area )
    {
        if ( area >= settings.minArea )
            areaRootVec.push_back( { area, root } );
    }

    // leave at most given number of roots sorted in descending by area order
    if ( areaRootVec.size() <= settings.maxLargeComponents )
    {
        if ( settings.numSmallerComponents )
            *settings.numSmallerComponents = 0;
        std::sort( areaRootVec.begin(), areaRootVec.end(), std::greater() );
    }
    else
    {
        if ( settings.numSmallerComponents )
            *settings.numSmallerComponents = int( areaRootVec.size() - settings.maxLargeComponents );
        std::partial_sort( areaRootVec.begin(), areaRootVec.begin() + settings.maxLargeComponents, areaRootVec.end(), std::greater() );
        areaRootVec.resize( settings.maxLargeComponents );
    }

    res.resize( areaRootVec.size() );
    ParallelFor( res, [&]( size_t i )
    {
        const auto myRoot = areaRootVec[i].root;
        auto & fs = res[i];
        fs.resize( mp.mesh.topology.faceSize() );
        for ( auto f : region )
            if ( roots[f] == myRoot )
                fs.set( f );
    } );
    return res;
}

VertBitSet getComponentsVerts( const Mesh& mesh, const VertBitSet& seeds, const VertBitSet* region /*= nullptr */ )
{
    MR_TIMER;

    VertBitSet res;
    if ( seeds.none() )
        return res;

    auto unionFindStruct = getUnionFindStructureVerts( mesh, region );
    const VertBitSet& vertRegion = mesh.topology.getVertIds( region );

    VertId vertRoot;
    for ( auto s : seeds )
    {
        if ( vertRoot < 0 )
            vertRoot = unionFindStruct.find( s );
        else
            vertRoot = unionFindStruct.unite( vertRoot, s ).first;
    }

    if ( vertRoot )
    {
        const auto& allRoots = unionFindStruct.roots();
        res.resize( allRoots.size() );
        BitSetParallelFor( vertRegion, [&]( VertId v )
        {
            if ( allRoots[v] == vertRoot )
                res.set( v );
        } );
    }
    return res;
}

size_t getNumComponents( const MeshPart& meshPart, FaceIncidence incidence, const UndirectedEdgeBitSet * isCompBd )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructureFaces( meshPart, incidence, isCompBd );
    const FaceBitSet& region = meshPart.mesh.topology.getFaceIds( meshPart.region );

    std::atomic<size_t> res{ 0 };
    tbb::parallel_for( tbb::blocked_range<FaceId>( 0_f, FaceId( unionFindStruct.size() ) ),
        [&]( const tbb::blocked_range<FaceId> & range )
    {
        size_t myRoots = 0;
        for ( auto f = range.begin(); f < range.end(); ++f )
        {
            if ( !region.test( f ) )
                continue;
            if ( f == unionFindStruct.findUpdateRange( f, range.begin(), range.end() ) )
                ++myRoots;
        }
        res.fetch_add( myRoots, std::memory_order_relaxed );
    } );
    return res;
}

std::pair<std::vector<FaceBitSet>, int> getAllComponents( const MeshPart& meshPart, int maxComponentCount,
    FaceIncidence incidence /*= FaceIncidence::PerEdge*/, const UndirectedEdgeBitSet * isCompBd /*= {}*/ )
{
    MR_TIMER;
    assert( maxComponentCount > 1 );
    if ( maxComponentCount < 1 )
        maxComponentCount = INT_MAX;
    const FaceBitSet& region = meshPart.mesh.topology.getFaceIds( meshPart.region );
    auto [uniqueRootsMap, componentsCount] = getAllComponentsMap( meshPart, incidence, isCompBd );
    if ( !componentsCount )
        return { {}, 0 };
    const int componentsInGroup = ( maxComponentCount == INT_MAX ) ? 1 : ( componentsCount + maxComponentCount - 1 ) / maxComponentCount;
    return { getAllComponents( uniqueRootsMap, componentsCount, region, maxComponentCount ), componentsInGroup };
}

std::vector<MR::FaceBitSet> getAllComponents( Face2RegionMap& componentsMap, int componentsCount, const FaceBitSet& region,
    int maxComponentCount )
{
    const int componentsInGroup = maxComponentCount == INT_MAX ? 1 : ( componentsCount + maxComponentCount - 1 ) / maxComponentCount;
    if ( componentsInGroup != 1 )
        for ( RegionId& id : componentsMap )
            id = RegionId( id / componentsInGroup );
    componentsCount = ( componentsCount + componentsInGroup - 1 ) / componentsInGroup;
    std::vector<FaceBitSet> res( componentsCount );
    // this block is needed to limit allocations for not packed meshes
    std::vector<int> resSizes( componentsCount, 0 );
    for ( auto f : region )
    {
        int index = componentsMap[f];
        if ( f > resSizes[index] )
            resSizes[index] = f;
    }
    for ( int i = 0; i < componentsCount; ++i )
        res[i].resize( resSizes[i] + 1 );
    // end of allocation block
    for ( auto f : region )
        res[componentsMap[f]].set( f );
    return res;
}

std::vector<MR::FaceBitSet> getAllComponents( const MeshPart& meshPart, FaceIncidence incidence /*= FaceIncidence::PerEdge*/,
    const UndirectedEdgeBitSet * isCompBd /*= {} */ )
{
    return getAllComponents( meshPart, INT_MAX, incidence, isCompBd ).first;
}

static void getUnionFindStructureFacesPerEdge( const MeshPart& meshPart, const UndirectedEdgeBitSet * isCompBd, UnionFind<FaceId>& res )
{
    MR_TIMER;

    const auto& mesh = meshPart.mesh;
    const FaceBitSet& region = mesh.topology.getFaceIds( meshPart.region );
    const FaceBitSet* lastPassFaces = &region;
    const auto numFaces = region.find_last() + 1;
    res.reset( numFaces );
    const auto numThreads = int( tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) );

    FaceBitSet bdFaces;
    if ( numThreads > 1 )
    {
        bdFaces.resize( numFaces );
        lastPassFaces = &bdFaces;

        BitSetParallelForAllRanged( region, [&] ( FaceId f0, const auto & range )
        {
            if ( !contains( region, f0 ) )
                return;
            EdgeId e[3];
            mesh.topology.getTriEdges( f0, e );
            for ( int i = 0; i < 3; ++i )
            {
                assert( mesh.topology.left( e[i] ) == f0 );
                FaceId f1 = mesh.topology.right( e[i] );
                if ( f0 < f1 && contains( meshPart.region, f1 ) )
                {
                    if ( f1 >= range.end )
                        bdFaces.set( f0 ); // remember the face to unite later in a sequential region
                    else if ( !isCompBd || !isCompBd->test( e[i].undirected() ) )
                        res.unite( f0, f1 ); // our region
                }
            }
        } );
    }

    for ( auto f0 : *lastPassFaces )
    {
        EdgeId e[3];
        mesh.topology.getTriEdges( f0, e );
        for ( int i = 0; i < 3; ++i )
        {
            assert( mesh.topology.left( e[i] ) == f0 );
            FaceId f1 = mesh.topology.right( e[i] );
            if ( f0 < f1 && contains( meshPart.region, f1 ) && ( !isCompBd || !isCompBd->test( e[i].undirected() ) ) )
                res.unite( f0, f1 );
        }
    }
}

std::pair<Face2RegionMap, int> getAllComponentsMap( const MeshPart& meshPart, FaceIncidence incidence, const UndirectedEdgeBitSet * isCompBd )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructureFaces( meshPart, incidence, isCompBd );
    const auto& mesh = meshPart.mesh;
    const FaceBitSet& region = mesh.topology.getFaceIds( meshPart.region );

    const auto& allRoots = unionFindStruct.roots();
    return getUniqueRootIds( allRoots, region );
}

Vector<double, RegionId> getRegionAreas( const MeshPart& meshPart,
    const Face2RegionMap & regionMap, int numRegions )
{
    MR_TIMER;
    Vector<double, RegionId> res( numRegions );
    for ( auto f : meshPart.mesh.topology.getFaceIds( meshPart.region ) )
        res[regionMap[f]] += meshPart.mesh.dblArea( f );

    for ( auto & a : res )
        a *= 0.5;

    return res;
}

std::pair<FaceBitSet, int> getLargeByAreaRegions( const MeshPart& meshPart,
    const Face2RegionMap & regionMap, int numRegions, float minArea )
{
    MR_TIMER;
    const auto regionAreas = getRegionAreas( meshPart, regionMap, numRegions );

    FaceBitSet largeRegions( meshPart.mesh.topology.faceSize() );
    BitSetParallelFor( meshPart.mesh.topology.getFaceIds( meshPart.region ), [&]( FaceId f )
    {
        if ( regionAreas[regionMap[f]] >= minArea )
            largeRegions.set( f );
    } );

    int numLargeRegions = 0;
    for ( const auto & a : regionAreas )
        if ( a >= minArea )
            ++numLargeRegions;

    return { std::move( largeRegions ), numLargeRegions };
}

static std::vector<VertBitSet> getAllComponentsVerts( UnionFind<VertId>& unionFindStruct, const VertBitSet& vertsRegion, const VertBitSet* doNotOutput )
{
    MR_TIMER;

    const auto& allRoots = unionFindStruct.roots();
    auto [uniqueRootsMap, k] = getUniqueRootIds( allRoots, vertsRegion );
    std::vector<VertBitSet> res( k, VertBitSet( allRoots.size() ) );
    for ( auto v : vertsRegion )
    {
        if ( doNotOutput && doNotOutput->test( v ) )
            continue;
        auto curRoot = allRoots[v];
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

std::vector<VertBitSet> getAllComponentsVertsSeparatedByPaths( const Mesh& mesh, const std::vector<SurfacePath>& paths )
{
    VertBitSet pathVerts;
    auto unionFindStruct = getUnionFindStructureVertsSeparatedByPaths( mesh, paths, &pathVerts );
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

std::vector<UndirectedEdgeBitSet> getAllComponentsUndirectedEdges( const Mesh& mesh, const UndirectedEdgeBitSet& edges )
{
    MR_TIMER;

    auto unionFindStruct = getUnionFindStructureVerts( mesh, edges );

    const auto& allRoots = unionFindStruct.roots();
    constexpr int InvalidRoot = -1;
    std::vector<int> uniqueRootsMap( allRoots.size(), InvalidRoot );
    int k = 0;

    for ( auto ue : edges )
    {
        const EdgeId e{ ue };
        int curRoot = allRoots[mesh.topology.org( e )];
        auto& uniqIndex = uniqueRootsMap[curRoot];
        if ( uniqIndex == InvalidRoot )
        {
            uniqIndex = k;
            ++k;
        }
    }
    std::vector<UndirectedEdgeBitSet> res( k, UndirectedEdgeBitSet( edges.size() ) );
    for ( auto ue : edges )
    {
        const EdgeId e{ ue };
        int curRoot = allRoots[mesh.topology.org( e )];
        res[uniqueRootsMap[curRoot]].set( ue );
    }
    return res;
}

bool hasFullySelectedComponent( const MeshTopology& topology, const VertBitSet & selection )
{
    MR_TIMER;

    auto unionFindStruct = getUnionFindStructureVerts( topology );
    const auto& allRoots = unionFindStruct.roots();
    auto [uniqueRootsMap, k] = getUniqueRootIds( allRoots, topology.getValidVerts() );
    RegionBitSet remainKeysBitSets( k );
    for ( VertId v( 0 ); v < uniqueRootsMap.size(); ++v )
    {
        if ( selection.test( v ) )
            continue;
        if ( auto rId = uniqueRootsMap[v] )
            remainKeysBitSets.set( rId );
    }
    return remainKeysBitSets.count() != remainKeysBitSets.size();
}

bool hasFullySelectedComponent( const Mesh& mesh, const VertBitSet & selection )
{
    return hasFullySelectedComponent( mesh.topology, selection );
}

void excludeFullySelectedComponents( const Mesh& mesh, VertBitSet& selection )
{
    MR_TIMER;

    auto unionFindStruct = getUnionFindStructureVerts( mesh );
    const auto& allRoots = unionFindStruct.roots();
    auto [uniqueRootsMap, k] = getUniqueRootIds( allRoots, mesh.topology.getValidVerts() );
    RegionBitSet remainKeysBitSets( k );
    for ( VertId v( 0 ); v < uniqueRootsMap.size(); ++v )
    {
        if ( selection.test( v ) )
            continue;
        if ( auto rId = uniqueRootsMap[v] )
            remainKeysBitSets.set( rId );
    }
    for ( auto v : selection )
    {
        if ( !remainKeysBitSets.test( uniqueRootsMap[v] ) )
            selection.reset( v );
    }
}

UnionFind<FaceId> getUnionFindStructureFacesPerEdge( const MeshPart& meshPart, const UndirectedEdgeBitSet * isCompBd )
{
    UnionFind<FaceId> res;
    getUnionFindStructureFacesPerEdge( meshPart, isCompBd, res );
    return res;
}

UnionFind<FaceId> getUnionFindStructureFaces( const MeshPart& meshPart, FaceIncidence incidence, const UndirectedEdgeBitSet * isCompBd )
{
    UnionFind<FaceId> res;
    if ( incidence == FaceIncidence::PerEdge )
        return getUnionFindStructureFacesPerEdge( meshPart, isCompBd );

    MR_TIMER;
    assert( !isCompBd );
    const auto& mesh = meshPart.mesh;
    const FaceBitSet& region = mesh.topology.getFaceIds( meshPart.region );
    res.reset( region.find_last() + 1 );
    assert ( incidence == FaceIncidence::PerVertex );
    VertBitSet store;
    for ( auto v : getIncidentVerts( mesh.topology, meshPart.region, store ) )
    {
        FaceId f0;
        for ( auto edge : orgRing( mesh.topology, v ) )
        {
            FaceId f1 = mesh.topology.left( edge );
            if ( !contains( meshPart.region, f1 ) )
                continue;
            if ( !f0 )
            {
                f0 = f1;
                continue;
            }
            res.unite( f0, f1 );
        }
    }
    return res;
}

UnionFind<VertId> getUnionFindStructureVerts( const MeshTopology& topology, const VertBitSet* region )
{
    MR_TIMER;

    const VertBitSet& vertsRegion = topology.getVertIds( region );

    auto test = [region]( VertId v )
    {
        if ( !region )
            return true;
        else
            return region->test( v );
    };

    static_assert( VertBitSet::npos + 1 == 0 );
    UnionFind<VertId> unionFindStructure( vertsRegion.find_last() + 1 );

    VertId v1;
    for ( auto v0 : vertsRegion )
    {
        for ( auto e : orgRing( topology, v0 ) )
        {
            v1 = topology.dest( e );
            if ( v1.valid() && test( v1 ) && v1 < v0 )
                unionFindStructure.unite( v0, v1 );
        }
    }
    return unionFindStructure;
}

UnionFind<VertId> getUnionFindStructureVerts( const Mesh& mesh, const VertBitSet* region )
{
    return getUnionFindStructureVerts( mesh.topology, region );
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

UnionFind<VertId> getUnionFindStructureVerts( const Mesh& mesh, const UndirectedEdgeBitSet& uEdges )
{
    MR_TIMER;

    UnionFind<VertId> unionFindStructure( mesh.topology.lastValidVert() + 1 );

    for ( EdgeId ue : uEdges )
    {
        auto vo = mesh.topology.org( ue );
        auto vd = mesh.topology.dest( ue );
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

UnionFind<VertId> getUnionFindStructureVertsSeparatedByPaths( const Mesh& mesh, const std::vector<SurfacePath>& paths, VertBitSet* outPathVerts )
{
    MR_TIMER;
    UndirectedEdgeBitSet ignoreEdges( mesh.topology.undirectedEdgeSize() );

    for ( const auto& path: paths )
        for ( const MeshEdgePoint& ep : path )
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

UnionFind<UndirectedEdgeId> getUnionFindStructureUndirectedEdges( const Mesh& mesh, bool allPointToRoots )
{
    MR_TIMER;

    UnionFind<UndirectedEdgeId> res( mesh.topology.undirectedEdgeSize() );
    const auto numThreads = int( tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) );

    UndirectedEdgeBitSet lastPass( mesh.topology.undirectedEdgeSize(), numThreads <= 1 );
    if ( numThreads > 1 )
    {
        BitSetParallelForAllRanged( lastPass, [&] ( UndirectedEdgeId ue, const auto & range )
        {
            const EdgeId e = ue;
            const UndirectedEdgeId ues[4] =
            {
                mesh.topology.prev( e ),
                mesh.topology.next( e ),
                mesh.topology.prev( e.sym() ),
                mesh.topology.next( e.sym() )
            };
            for ( int i = 0; i < 4; ++i )
            {
                const auto uei = ues[i];
                if ( ue < uei )
                {
                    if ( uei >= range.end )
                        lastPass.set( ue ); // remember the edge to unite later in a sequential region
                    else
                        res.unite( ue, uei ); // our region
                }
            }
        } );
    }

    for ( auto ue : lastPass )
    {
        const EdgeId e = ue;
        const UndirectedEdgeId ues[4] =
        {
            mesh.topology.prev( e ),
            mesh.topology.next( e ),
            mesh.topology.prev( e.sym() ),
            mesh.topology.next( e.sym() )
        };
        for ( int i = 0; i < 4; ++i )
        {
            const auto uei = ues[i];
            if ( ue < uei )
                res.unite( ue, uei );
        }
    }

    if ( allPointToRoots )
    {
        tbb::parallel_for( tbb::blocked_range( 0_ue, UndirectedEdgeId( res.size() ) ),
            [&] ( const tbb::blocked_range<UndirectedEdgeId>& range )
        {
            for ( UndirectedEdgeId ue = range.begin(); ue < range.end(); ++ue )
                res.findUpdateRange( ue, range.begin(), range.end() );
        } );
    }

    return res;
}

UndirectedEdgeBitSet getComponentsUndirectedEdges( const Mesh& mesh, const UndirectedEdgeBitSet& seeds )
{
    MR_TIMER;
    auto unionFindStruct = getUnionFindStructureUndirectedEdges( mesh, true );

    UndirectedEdgeId commonRoot;
    for ( auto s : seeds )
    {
        if ( commonRoot )
            commonRoot = unionFindStruct.unite( commonRoot, s ).first;
        else
            commonRoot = unionFindStruct.find( s );
    }

    UndirectedEdgeBitSet res;
    if ( commonRoot )
    {
        const auto& allRoots = unionFindStruct.roots();
        res.resize( allRoots.size() );
        BitSetParallelForAll( res, [&]( UndirectedEdgeId ue )
        {
            if ( allRoots[ue] == commonRoot )
                res.set( ue );
        } );
    }
    return res;
}

} // namespace MeshComponents

} // namespace MR
