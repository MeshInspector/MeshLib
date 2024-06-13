#include "MRFillHoleNicely.h"
#include "MRMesh.h"
#include "MRMeshFillHole.h"
#include "MRTimer.h"

namespace MR
{

FaceBitSet fillHoleNicely( Mesh & mesh,
    EdgeId holeEdge,
    const FillHoleNicelySettings & settings )
{
    MR_TIMER
    assert( !mesh.topology.left( holeEdge ) );

    FaceBitSet res;
    if ( mesh.topology.left( holeEdge ) )
        return res;

    const auto fsz0 = mesh.topology.faceSize();
    fillHole( mesh, holeEdge, {
            .metric = getMetrics_()[int( settings_.currMetric ) ]( mesh,initEdge ),
            .multipleEdgesResolveMode = FillHoleParams::MultipleEdgesResolveMode::Strong,
            .maxPolygonSubdivisions = settings_.maxPolygonSubdivisions } );
    FaceBitSet newFaces;
    const auto fsz = mesh.topology.faceSize();
    if ( fsz0 == fsz )
        return newFaces;
    newFaces.autoResizeSet( FaceId{ fsz0 }, fsz - fsz0 );

    if ( !settings_.triangulateOnly )
    {
        VertBitSet newVerts;
        SubdivideSettings settings;
        settings.maxEdgeLen = settings_.maxEdgeLen;
        settings.maxEdgeSplits = 20000;
        settings.region = &newFaces;
        settings.newVerts = &newVerts;

        const auto lastVert = mesh.topology.lastValidVert();
        bool updateUV = uvCoords && lastVert < uvCoords->size();
        bool updateColorMap = colorMap && lastVert < colorMap->size();
        VertUVCoords uv;
        VertColors cm;
        VertUVCoords& uvCoordsRef = uvCoords ? *uvCoords : uv;
        VertColors& colorMapRef = colorMap ? *colorMap : cm;
        if ( updateUV || updateColorMap )
        {
            settings.onEdgeSplit = [&mesh, &uvCoordsRef, &colorMapRef, updateUV, updateColorMap] ( EdgeId e1, EdgeId e )
            {
                const auto org = mesh.topology.org( e1 );
                const auto dest = mesh.topology.dest( e );

                if ( updateUV )
                    uvCoordsRef.push_back( ( uvCoordsRef[org] + uvCoordsRef[dest] ) * 0.5f );

                if ( updateColorMap )
                    colorMapRef.push_back( colorMapRef[org] + ( colorMapRef[dest] - colorMapRef[org] ) * 0.5f );
            };
        }

        subdivideMesh( mesh, settings );

        if ( settings_.smoothCurvature )
            positionVertsSmoothly( mesh, newVerts );
    }


    return res;
}

} //namespace MR
