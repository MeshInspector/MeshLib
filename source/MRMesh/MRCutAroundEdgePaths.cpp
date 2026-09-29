#include "MRCutAroundEdgePaths.h"
#include "MRSurfaceDistance.h"
#include "MRExtractIsolines.h"
#include "MROneMeshContours.h"
#include "MRContoursCut.h"
#include "MRFillContour.h"
#include "MRMeshComponents.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

Expected<std::vector<FaceBitSet>> cutAroundEdgePaths( Mesh & mesh, const std::vector<EdgePath> & paths,
    const CutAroundEdgePathsParams & params )
{
    MR_TIMER;
    if ( !( params.distance > 0 ) )
        return unexpected( "distance must be positive" );
    if ( !( params.minSpacing >= 0 ) )
        return unexpected( "minSpacing must not be negative" );

    const auto numPaths = paths.size();
    std::vector<VertScalars> dists( numPaths );
    ParallelFor( dists, [&]( size_t i )
    {
        if ( !paths[i].empty() )
        {
            VertBitSet starts( mesh.topology.vertSize() );
            for ( auto e : paths[i] )
            {
                starts.set( mesh.topology.org( e ) );
                starts.set( mesh.topology.dest( e ) );
            }
            // values in (distance, distance + minSpacing] are needed to keep minSpacing from the region of another path
            dists[i] = computeSurfaceDistances( mesh, starts, params.distance + params.minSpacing );
        }
        dists[i].resize( mesh.topology.vertSize(), FLT_MAX );
    } );

    const float sumSpacing = 2 * params.distance + params.minSpacing;
    if ( numPaths > 1 )
    {
        BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
        {
            for ( size_t i = 0; i + 1 < numPaths; ++i )
            {
                auto & di = dists[i][v];
                if ( di == FLT_MAX )
                    continue;
                for ( size_t j = i + 1; j < numPaths; ++j )
                {
                    auto & dj = dists[j][v];
                    if ( dj == FLT_MAX )
                        continue;
                    const auto sum = di + dj;
                    if ( sum >= sumSpacing )
                        continue;
                    if ( sum > 0 )
                    {
                        const auto k = sumSpacing / sum;
                        di *= k;
                        dj *= k;
                    }
                    else
                        di = dj = sumSpacing / 2;
                }
            }
        } );
    }

    std::vector<IsoLines> isolines( numPaths );
    ParallelFor( isolines, [&]( size_t i )
    {
        isolines[i] = extractIsolines( mesh.topology, dists[i], params.distance );
    } );
    dists = {};

    IsoLines allIsolines;
    std::vector<size_t> firstIsoline( numPaths + 1 );
    for ( size_t i = 0; i < numPaths; ++i )
    {
        firstIsoline[i] = allIsolines.size();
        for ( auto & isoline : isolines[i] )
            allIsolines.push_back( std::move( isoline ) );
    }
    firstIsoline[numPaths] = allIsolines.size();

    const auto cutRes = cutMesh( mesh, convertSurfacePathsToMeshContours( mesh, allIsolines ) );
    if ( cutRes.fbsWithContourIntersections.any() )
        return unexpected( "isolines around the paths intersect" );
    assert( cutRes.resultCut.size() == allIsolines.size() );

    std::vector<FaceBitSet> res( numPaths );
    ParallelFor( res, [&]( size_t i )
    {
        if ( paths[i].empty() )
            return;
        if ( firstIsoline[i] < firstIsoline[i + 1] )
        {
            // the isolines are directed to have smaller distances on the left
            const std::vector<EdgePath> cuts( cutRes.resultCut.begin() + firstIsoline[i], cutRes.resultCut.begin() + firstIsoline[i + 1] );
            res[i] = fillContourLeft( mesh.topology, cuts );
        }
        else
        {
            // no isoline: the whole connected component is closer than distance to the path
            FaceBitSet seeds( mesh.topology.faceSize() );
            for ( auto e : paths[i] )
            {
                if ( auto l = mesh.topology.left( e ) )
                    seeds.set( l );
                if ( auto r = mesh.topology.right( e ) )
                    seeds.set( r );
            }
            res[i] = MeshComponents::getComponents( mesh, seeds );
        }
    } );
    return res;
}

} //namespace MR
