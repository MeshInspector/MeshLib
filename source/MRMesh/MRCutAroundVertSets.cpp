#include "MRCutAroundVertSets.h"
#include "MRSurfaceDistance.h"
#include "MRExtractIsolines.h"
#include "MROneMeshContours.h"
#include "MRContoursCut.h"
#include "MRFillContour.h"
#include "MRMeshComponents.h"
#include "MRRegionBoundary.h"
#include "MRMesh.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRFmt.h"

namespace MR
{

Expected<std::vector<FaceBitSet>> cutAroundVertSets( Mesh & mesh, const std::vector<VertBitSet> & vertSets,
    const CutAroundVertSetsParams & params, const ProgressCallback & cb )
{
    MR_TIMER;
    if ( !( params.distance > 0 ) )
        return unexpected( "distance must be positive" );
    if ( !( params.gap > 0 ) )
        return unexpected( "gap must be positive" );

    const auto numSets = vertSets.size();
    VertBitSet allVerts;
    for ( size_t i = 0; i < numSets; ++i )
    {
        if ( allVerts.intersects( vertSets[i] ) )
            return unexpected( fmt::format( "vertex set #{} shares a vertex with a previous set", i ) );
        allVerts |= vertSets[i];
    }

    auto [distCb, adjustCb, isolinesCb, cutCb, regionsCb] = splitProgress( cb, 0.5f, 0.55f, 0.6f, 0.9f );

    std::vector<VertScalars> dists( numSets );
    if ( !ParallelFor( dists, [&]( size_t i )
    {
        if ( vertSets[i].any() )
        {
            // values in (distance, distance + gap] are needed to keep the gap from the region of another set
            dists[i] = computeSurfaceDistances( mesh, vertSets[i], params.distance + params.gap );
        }
        dists[i].resize( mesh.topology.vertSize(), FLT_MAX );
    }, distCb, 1 ) )
        return unexpectedOperationCanceled();

    const float minSum = 2 * params.distance + params.gap;
    if ( numSets > 1 && !BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        for ( size_t i = 0; i + 1 < numSets; ++i )
        {
            auto & di = dists[i][v];
            if ( di == FLT_MAX )
                continue;
            for ( size_t j = i + 1; j < numSets; ++j )
            {
                auto & dj = dists[j][v];
                if ( dj == FLT_MAX )
                    continue;
                const auto sum = di + dj;
                if ( sum >= minSum )
                    continue;
                if ( sum > 0 )
                {
                    const auto k = minSum / sum;
                    di *= k;
                    dj *= k;
                }
                else
                    di = dj = minSum / 2;
            }
        }
    }, adjustCb ) )
        return unexpectedOperationCanceled();

    std::vector<IsoLines> isolines( numSets );
    if ( !ParallelFor( isolines, [&]( size_t i )
    {
        isolines[i] = extractIsolines( mesh.topology, dists[i], params.distance );
    }, isolinesCb, 1 ) )
        return unexpectedOperationCanceled();
    dists = {};

    IsoLines allIsolines;
    std::vector<size_t> firstIsoline( numSets + 1 );
    for ( size_t i = 0; i < numSets; ++i )
    {
        firstIsoline[i] = allIsolines.size();
        for ( auto & isoline : isolines[i] )
            allIsolines.push_back( std::move( isoline ) );
    }
    firstIsoline[numSets] = allIsolines.size();

    const auto cutRes = cutMesh( mesh, convertSurfacePathsToMeshContours( mesh, allIsolines ) );
    if ( cutRes.fbsWithContourIntersections.any() )
        return unexpected( "isolines around the vertex sets intersect" );
    assert( cutRes.resultCut.size() == allIsolines.size() );
    if ( !reportProgress( cutCb, 1.0f ) )
        return unexpectedOperationCanceled();

    std::vector<FaceBitSet> res( numSets );
    if ( !ParallelFor( res, [&]( size_t i )
    {
        if ( vertSets[i].none() )
            return;
        if ( firstIsoline[i] < firstIsoline[i + 1] )
        {
            // the isolines are directed to have smaller distances on the left
            const std::vector<EdgePath> cuts( cutRes.resultCut.begin() + firstIsoline[i], cutRes.resultCut.begin() + firstIsoline[i + 1] );
            res[i] = fillContourLeft( mesh.topology, cuts );
        }
        else
        {
            // no isoline: the whole connected component is closer than distance to the set
            res[i] = MeshComponents::getComponents( mesh, getIncidentFaces( mesh.topology, vertSets[i] ) );
        }
    }, regionsCb, 1 ) )
        return unexpectedOperationCanceled();
    return res;
}

} //namespace MR
