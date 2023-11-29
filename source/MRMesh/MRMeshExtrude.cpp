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

    auto& topology = mesh.topology;
    if ( region.none() || ( topology.getValidFaces() - region ).none() )
        return;

    if ( params.new2OldMap )
    {
        params.new2OldMap->resize( mesh.points.size() );
        ParallelFor( *params.new2OldMap, [&] ( VertId v )
        {
            params.new2OldMap->operator[]( v ) = v;
        } );
    }

    float maxEdgeLenSq = 0;

    auto componentBoundary = findLeftBoundaryInsideMesh( topology, region );
    for ( auto& contour : componentBoundary )
    {
        auto newContour = cutAlongEdgeLoop( mesh, contour );
        auto newEdge = makeDegenerateBandAroundHole( mesh, contour[0], params.outNewFaces );
        auto holeContour = trackRightBoundaryLoop( topology, newEdge );       

        if ( params.outExtrudedEdges || params.new2OldMap || params.maxEdgeLength )
        {
            if ( params.new2OldMap )
                params.new2OldMap->reserve( mesh.points.size() );

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
                    params.new2OldMap->autoResizeSet( mesh.topology.org( newContour[i] ), params.new2OldMap->operator[]( mesh.topology.org( contour[i] ) ) );
                    params.new2OldMap->autoResizeSet( mesh.topology.org( holeContour[i] ), params.new2OldMap->operator[]( mesh.topology.org( contour[i] ) ) );
                }
            }
        }

        stitchContours( topology, holeContour, newContour );
    }

    if ( params.maxEdgeLength )
        *params.maxEdgeLength = sqrt( maxEdgeLenSq );
}

} // namespace MR
