#include "MRMeshExtrude.h"
#include "MRMesh.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"
#include "MRContoursStitch.h"
#include "MRMeshFillHole.h"
#include "MRColor.h"
#include "MRVector2.h"
#include "MRParallelFor.h"

namespace MR
{

void makeDegenerateBandAroundRegion( Mesh& mesh, const FaceBitSet& region, const MakeDegenerateBandAroundRegionParams& params )
{
    MR_TIMER
    MR_WRITER( mesh )
    if ( region.none() )
            return;

    auto& topology = mesh.topology;

    float maxEdgeLenSq = 0;

    auto componentBoundary = findLeftBoundary( topology, region );
    for ( auto& contour : componentBoundary )
    {
        auto newContour = cutAlongEdgeLoop( mesh, contour );
        auto newEdge = makeDegenerateBandAroundHole( mesh, contour[0], params.outNewFaces );
        auto holeContour = trackRightBoundaryLoop( topology, newEdge );

        if ( params.outExtrudedEdges || params.new2OldMap || params.maxEdgeLength )
        {
            for ( size_t i = 0; i < contour.size(); ++i )
            {
                maxEdgeLenSq = std::max( mesh.edgeLengthSq( contour[i] ), maxEdgeLenSq );

                if ( params.outExtrudedEdges )
                {
                    const auto ue = topology.findEdge( topology.org( contour[i] ), topology.org( holeContour[i] ) ).undirected();
                    if ( ue.valid() )
                        params.outExtrudedEdges->autoResizeSet( ue, true );
                }

                if ( params.new2OldMap )
                {
                    ( *params.new2OldMap )[mesh.topology.org( newContour[i] )] = mesh.topology.org( contour[i] );
                    ( *params.new2OldMap )[mesh.topology.org( holeContour[i] )] = mesh.topology.org( contour[i] );
                }
            }
        }

        stitchContours( topology, holeContour, newContour );
    }

    if ( params.maxEdgeLength )
        *params.maxEdgeLength = sqrt( maxEdgeLenSq );
}

} // namespace MR
