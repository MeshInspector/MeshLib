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

static void smoothFillingNicely( Mesh& mesh, const VertBitSet& newVerts, const FaceBitSet& newFaces, bool smoothBd, const SmoothFillingSettings& settings )
{
    // exclude boundary vertices from positionVertsSmoothly(), since it tends to move them inside the mesh
    auto vertsForSmoothing = newVerts - mesh.topology.findBdVerts( nullptr, &newVerts );
    positionVertsSmoothlySharpBd( mesh, { .region = &vertsForSmoothing } );
    if ( !smoothBd )
        return;
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
            positionVertsSmoothlySharpBd( mesh, { .region = &vertsForSmoothing } );
            positionVertsSmoothly( mesh, vertsForSmoothing, settings.edgeWeights, settings.vmass );
        }
    }
}

static std::pair<FaceColors*, Color> prepareFillingFaceColors( const MeshTopology& tp, EdgeId h0, EdgeId h1, FaceColors* inFaceColors )
{
    FaceColors* faceColors = inFaceColors && tp.lastValidFace() < inFaceColors->size() ?
        inFaceColors : nullptr;

    Color newFaceColor;
    if ( faceColors )
    {
        //compute average color of faces around the hole
        Vector4i sumColors;
        int num = 0;
        if ( h0 )
        {
            for ( auto e : leftRing( tp, h0) )
            {
                auto r = tp.right( e );
                if ( !r )
                    continue;
                sumColors += Vector4i( ( *faceColors )[r] );
                ++num;
            }
        }
        if ( h1 )
        {
            for ( auto e : leftRing( tp, h1 ) )
            {
                auto r = tp.right( e );
                if ( !r )
                    continue;
                sumColors += Vector4i( ( *faceColors )[r] );
                ++num;
            }
        }
        if ( num > 0 )
            newFaceColor = Color( sumColors / num );
    }
    return std::make_pair( faceColors, newFaceColor );
}

static VertBitSet subdivideFillingNicely( Mesh& mesh, FaceBitSet& newFaces, 
    const SubdivideFillingSettings& settings, const OutAttributesFillingSettings& outAttribs )
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
    VertUVCoords* uvCoords = outAttribs.uvCoords && lastVert < outAttribs.uvCoords->size() ?
        outAttribs.uvCoords : nullptr;
    VertColors* colorMap = outAttribs.colorMap && lastVert < outAttribs.colorMap->size() ?
        outAttribs.colorMap : nullptr;
    if ( uvCoords || colorMap || outAttribs.faceColors )
    {
        subset.onEdgeSplit = [&mesh, uvCoords, colorMap, faceColors = outAttribs.faceColors] ( EdgeId e1, EdgeId e )
        {
            const auto org = mesh.topology.org( e1 );
            const auto dest = mesh.topology.dest( e );
            const auto newV = mesh.topology.org( e );
            assert( newV == mesh.topology.dest( e1 ) );

            if ( uvCoords )
                uvCoords->autoResizeSet( newV, ( ( *uvCoords )[org] + ( *uvCoords )[dest] ) * 0.5f );

            if ( colorMap )
                colorMap->autoResizeSet( newV, ( *colorMap )[org] + ( ( *colorMap )[dest] - ( *colorMap )[org] ) * 0.5f );

            if ( faceColors )
            {
                if ( auto l = mesh.topology.left( e ) )
                {
                    auto l1 = mesh.topology.left( e1 );
                    assert( l1 && l < l1 );
                    faceColors->autoResizeSet( l1, ( *faceColors )[l] );
                }
                if ( auto r = mesh.topology.right( e ) )
                {
                    auto r1 = mesh.topology.right( e1 );
                    assert( r1 && r < r1 );
                    faceColors->autoResizeSet( r1, ( *faceColors )[r] );
                }
            }
        };
    }

    subdivideMesh( mesh, subset );
    return newVerts;
}

FaceBitSet fillHoleNicely( Mesh & mesh,
    EdgeId holeEdge,
    const FillHoleNicelySettings & settings )
{
    MR_TIMER;
    assert( !mesh.topology.left( holeEdge ) );

    FaceBitSet newFaces;
    if ( mesh.topology.left( holeEdge ) )
        return newFaces; //no hole exists

    auto [faceColors, newFaceColor] = prepareFillingFaceColors( mesh.topology, holeEdge, EdgeId(), settings.outAttributes.faceColors );

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
        auto outAttribs = settings.outAttributes;
        outAttribs.faceColors = faceColors;
        VertBitSet newVerts = subdivideFillingNicely( mesh, newFaces, settings.subdivideSettings, outAttribs );

        if ( settings.smoothCurvature )
        {
            smoothFillingNicely( mesh, newVerts, newFaces, settings.triangulateParams.smoothBd, settings.smoothSeettings );
        }
    }

    return newFaces;
}

} //namespace MR
