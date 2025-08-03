#include "MRFillHoleNicely.h"
#include "MRMesh.h"
#include "MRMeshSubdivide.h"
#include "MRVector2.h"
#include "MRColor.h"
#include "MRPositionVertsSmoothly.h"
#include "MRTimer.h"
#include "MRRegionBoundary.h"
#include "MRExpandShrink.h"
#include "MRMeshComponents.h"
#include "MRRingIterator.h"

namespace MR
{

FaceBitSet fillHoleNicely( Mesh & mesh,
    EdgeId holeEdge,
    const FillHoleNicelySettings & settings )
{
    MR_TIMER;
    assert( !mesh.topology.left( holeEdge ) );

    FaceBitSet newFaces;
    if ( mesh.topology.left( holeEdge ) )
        return newFaces; //no hole exists

    FaceColors * const faceColors = settings.faceColors && mesh.topology.lastValidFace() < settings.faceColors->size() ? settings.faceColors : nullptr;

    Color newFaceColor;
    if ( faceColors )
    {
        //compute average color of faces around the hole
        Vector4i sumColors;
        int num = 0;
        for ( auto e : leftRing( mesh.topology, holeEdge ) )
        {
            auto r = mesh.topology.right( e );
            if ( !r )
                continue;
            sumColors += Vector4i( (*faceColors)[r] );
            ++num;
        }
        if ( num > 0 )
            newFaceColor = Color( sumColors / num );
    }

    const auto fsz0 = mesh.topology.faceSize();
    fillHole( mesh, holeEdge, settings.triangulateParams );
    const auto fsz = mesh.topology.faceSize();
    if ( fsz0 == fsz )
        return newFaces;
    newFaces.autoResizeSet( FaceId{ fsz0 }, fsz - fsz0 );
    if ( faceColors )
        faceColors->autoResizeSet( FaceId{ fsz0 }, fsz - fsz0, newFaceColor );

    if ( !settings.triangulateOnly )
    {
        VertBitSet newVerts;
        SubdivideSettings subset
        {
            .maxEdgeLen = settings.maxEdgeLen,
            .maxEdgeSplits = settings.maxEdgeSplits,
            .maxAngleChangeAfterFlip = settings.maxAngleChangeAfterFlip,
            .region = &newFaces,
            .notFlippable = settings.notFlippable,
            .newVerts = &newVerts,
            .beforeEdgeSplit = settings.beforeEdgeSplit,
            .onEdgeSplit = settings.onEdgeSplit
        };

        const auto lastVert = mesh.topology.lastValidVert();
        VertUVCoords * uvCoords = settings.uvCoords && lastVert < settings.uvCoords->size() ? settings.uvCoords : nullptr;
        VertColors * colorMap = settings.colorMap && lastVert < settings.colorMap->size() ? settings.colorMap : nullptr;
        if ( uvCoords || colorMap || faceColors )
        {
            subset.onEdgeSplit = [&mesh, uvCoords, colorMap, faceColors] ( EdgeId e1, EdgeId e )
            {
                const auto org = mesh.topology.org( e1 );
                const auto dest = mesh.topology.dest( e );
                const auto newV = mesh.topology.org( e );
                assert( newV == mesh.topology.dest( e1 ) );

                if ( uvCoords )
                    uvCoords->autoResizeSet( newV, ( (*uvCoords)[org] + (*uvCoords)[dest] ) * 0.5f );

                if ( colorMap )
                    colorMap->autoResizeSet( newV, (*colorMap)[org] + ( (*colorMap)[dest] - (*colorMap)[org] ) * 0.5f );

                if ( faceColors )
                {
                    if ( auto l = mesh.topology.left( e ) )
                    {
                        auto l1 = mesh.topology.left( e1 );
                        assert( l1 && l < l1 );
                        faceColors->autoResizeSet( l1, (*faceColors)[l] );
                    }
                    if ( auto r = mesh.topology.right( e ) )
                    {
                        auto r1 = mesh.topology.right( e1 );
                        assert( r1 && r < r1 );
                        faceColors->autoResizeSet( r1, (*faceColors)[r] );
                    }
                }
            };
        }

        subdivideMesh( mesh, subset );

        if ( settings.smoothCurvature )
        {
            // exclude boundary vertices from positionVertsSmoothly(), since it tends to move them inside the mesh
            auto vertsForSmoothing = newVerts - mesh.topology.findBdVerts( nullptr, &newVerts );
            positionVertsSmoothlySharpBd( mesh, vertsForSmoothing );
            if ( settings.triangulateParams.smoothBd )
            {
                positionVertsSmoothly( mesh, vertsForSmoothing, settings.edgeWeights, settings.vmass );
                if ( settings.naturalSmooth )
                {
                    auto undirectedEdgeBitSet = findRegionBoundaryUndirectedEdgesInsideMesh( mesh.topology, newFaces );
                    auto incidentVerts = getIncidentVerts( mesh.topology, undirectedEdgeBitSet );
                    expand( mesh.topology, incidentVerts, 5 );
                    shrink( mesh.topology, incidentVerts, 2 );
                    MeshComponents::excludeFullySelectedComponents( mesh, incidentVerts );
                    if ( incidentVerts.any() )
                    {
                        vertsForSmoothing = incidentVerts - mesh.topology.findBdVerts( nullptr, &incidentVerts );
                        positionVertsSmoothlySharpBd( mesh, vertsForSmoothing );
                        positionVertsSmoothly( mesh, vertsForSmoothing, settings.edgeWeights, settings.vmass );
                    }
                }
            }
        }

    }

    return newFaces;
}

} //namespace MR
