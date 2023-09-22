#include "MRMeshExtrude.h"
#include "MRMesh.h"
#include "MRRegionBoundary.h"
#include "MRMapEdge.h"
#include "MRTimer.h"

namespace MR
{

void makeDegenerateBandAroundRegion( Mesh& mesh, const FaceBitSet& region, FaceBitSet* outNewFaces, FaceHashMap* old2newMap )
{
    MR_TIMER
    MR_WRITER( mesh )

    auto& topology = mesh.topology;
    if ( region.none() || ( topology.getValidFaces() - region ).none() )
        return;

    auto componentBoundary = findLeftBoundaryInsideMesh( topology, region );
    auto meshBoundary = componentBoundary;

    FaceHashMap src2tgtFaces;
    WholeEdgeHashMap src2tgtEdges;
    PartMapping mapping {
        .src2tgtFaces = old2newMap ? &src2tgtFaces : nullptr,
        .src2tgtEdges = &src2tgtEdges,
    };
    auto component = mesh.cloneRegion( region, false, mapping );

    size_t boundaryEdgeCount = 0;
    for ( auto& contour : componentBoundary )
    {
        boundaryEdgeCount += contour.size();
        for ( auto& be : contour )
        {
            be = mapEdge( src2tgtEdges, be );
            assert( component.topology.isLeftBdEdge( be ) );
        }
    }

    topology.deleteFaces( region );
#ifndef NDEBUG
    for ( const auto& contour : meshBoundary )
        for ( const auto be : contour )
            assert( topology.isBdEdge( be ) );
#endif

    FaceMap outFmap;
    WholeEdgeMap outEmap;
    mesh.addPart( component, old2newMap ? &outFmap : nullptr, nullptr, &outEmap );

    for ( auto& contour : componentBoundary )
    {
        for ( auto& be : contour )
        {
            be = mapEdge( outEmap, be );
            assert( topology.isLeftBdEdge( be ) );
        }
    }

    if ( old2newMap )
    {
        old2newMap->reserve( topology.lastValidFace() + 1 );
        for ( const auto f : region )
            ( *old2newMap )[f] = outFmap[src2tgtFaces[f]];
    }

    topology.edgeReserve( topology.edgeSize() + boundaryEdgeCount * 4 );
    topology.faceReserve( topology.faceSize() + boundaryEdgeCount * 2 );
    if ( outNewFaces )
        outNewFaces->resize( topology.faceSize() + boundaryEdgeCount * 2 );

    for ( auto ci = 0u; ci < meshBoundary.size(); ++ci )
    {
        const auto& meshContour = meshBoundary[ci];
        const auto& componentContour = componentBoundary[ci];

        const auto closedLoop = topology.dest( meshContour.back() ) == topology.org( meshContour.front() );

        for ( auto ei = 0u; ei < meshContour.size(); ++ei )
        {
            const auto mbe = meshContour[ei];
            const auto cbe = componentContour[ei];

            if ( ei == 0 )
            {
                const auto cbe0 = topology.prev( cbe ).sym();
                assert( topology.isBdEdge( cbe0.sym() ) );

                const auto ne0 = topology.makeEdge();
                topology.splice( mbe, ne0 );
                topology.splice( cbe0.sym(), ne0.sym() );
            }

            const auto ne = topology.makeEdge();
            topology.splice( mbe, ne );
            topology.splice( cbe.sym(), ne.sym() );

            if ( ei < meshContour.size() - 1 || !closedLoop )
            {
                const auto mbe1 = mesh.topology.prev( mbe.sym() );
                assert( mesh.topology.isBdEdge( mbe1 ) );

                const auto ne1 = topology.makeEdge();
                topology.splice( mbe1, ne1 );
                topology.splice( ne.sym(), ne1.sym() );
            }

            const auto f0 = mesh.topology.addFaceId();
            const auto f1 = mesh.topology.addFaceId();
            mesh.topology.setLeft( ne, f0 );
            mesh.topology.setLeft( ne.sym(), f1 );
            if ( outNewFaces )
            {
                ( *outNewFaces ).set( f0 );
                ( *outNewFaces ).set( f1 );
            }
        }
    }
}

} // namespace MR
