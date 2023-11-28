#include "MRMeshExtrude.h"
#include "MRMesh.h"
#include "MRRegionBoundary.h"
#include "MRTimer.h"
#include "MRContoursStitch.h"
#include "MRMeshFillHole.h"

namespace MR
{

void makeDegenerateBandAroundRegion( Mesh& mesh, const ExtrudeParams& params )
{
    MR_TIMER
    MR_WRITER( mesh )

    auto& topology = mesh.topology;
    if ( params.region.none() || ( topology.getValidFaces() - params.region ).none() )
        return;

    float maxEdgeLenSq = 0;

    auto componentBoundary = findLeftBoundaryInsideMesh( topology, params.region );
    for ( auto& contour : componentBoundary )
    {
        auto newContour = cutAlongEdgeLoop( mesh, contour );
        auto newEdge = makeDegenerateBandAroundHole( mesh, contour[0], params.outNewFaces );
        auto holeContour = trackRightBoundaryLoop( topology, newEdge );

        stitchContours( topology, holeContour, newContour );
        if ( !params.outExtrudedEdges )
            continue;

        for ( size_t i = 0; i < contour.size(); ++i )
        {
            maxEdgeLenSq = std::max( mesh.edgeLengthSq( contour[i] ), maxEdgeLenSq );

            const auto ue = topology.findEdge( topology.org( contour[i] ), topology.org( holeContour[i] ) ).undirected();
            if ( ue.valid() )
                params.outExtrudedEdges->autoResizeSet( ue, true );
        }
    }

    if ( params.maxEdgeLength )
        *params.maxEdgeLength = sqrt( maxEdgeLenSq );
}

} // namespace MR
