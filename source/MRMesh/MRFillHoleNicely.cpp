#include "MRFillHoleNicely.h"
#include "MRMesh.h"
#include "MRMeshFillHole.h"
#include "MRMeshSubdivide.h"
#include "MRVector2.h"
#include "MRColor.h"
#include "MRPositionVertsSmoothly.h"
#include "MRTimer.h"

namespace MR
{

FaceBitSet fillHoleNicely( Mesh & mesh,
    EdgeId holeEdge,
    const FillHoleNicelySettings & settings )
{
    MR_TIMER
    assert( !mesh.topology.left( holeEdge ) );

    FaceBitSet newFaces;
    if ( mesh.topology.left( holeEdge ) )
        return newFaces;

    const auto fsz0 = mesh.topology.faceSize();
    fillHole( mesh, holeEdge,
        {
            .metric = settings.triangulateMetric,
            .multipleEdgesResolveMode = FillHoleParams::MultipleEdgesResolveMode::Strong,
            .maxPolygonSubdivisions = settings.maxPolygonSubdivisions
        } );
    const auto fsz = mesh.topology.faceSize();
    if ( fsz0 == fsz )
        return newFaces;
    newFaces.autoResizeSet( FaceId{ fsz0 }, fsz - fsz0 );

    if ( !settings.triangulateOnly )
    {
        VertBitSet newVerts;
        SubdivideSettings subset;
        subset.maxEdgeLen = settings.maxEdgeLen;
        subset.maxEdgeSplits = settings.maxEdgeSplits;
        subset.region = &newFaces;
        subset.newVerts = &newVerts;

        const auto lastVert = mesh.topology.lastValidVert();
        VertUVCoords * uvCoords = settings.uvCoords && lastVert < settings.uvCoords->size() ? settings.uvCoords : nullptr;
        VertColors * colorMap = settings.colorMap && lastVert < settings.colorMap->size() ? settings.colorMap : nullptr;
        if ( uvCoords || colorMap )
        {
            subset.onEdgeSplit = [&mesh, uvCoords, colorMap] ( EdgeId e1, EdgeId e )
            {
                const auto org = mesh.topology.org( e1 );
                const auto dest = mesh.topology.dest( e );

                if ( uvCoords )
                    uvCoords->push_back( ( (*uvCoords)[org] + (*uvCoords)[dest] ) * 0.5f );

                if ( colorMap )
                    colorMap->push_back( (*colorMap)[org] + ( (*colorMap)[dest] - (*colorMap)[org] ) * 0.5f );
            };
        }

        subdivideMesh( mesh, subset );

        if ( settings.smoothCurvature )
            positionVertsSmoothly( mesh, newVerts );
    }

    return newFaces;
}

} //namespace MR
