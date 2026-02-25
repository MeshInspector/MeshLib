#include "MRMeshDivideWithPlane.h"
#include "MRTimer.h"
#include "MRMesh.h"
#include "MRMeshTrimWithPlane.h"
#include "MRFillContours2D.h"
#include "MRMeshSubdivide.h"

namespace MR
{

void divideMeshWithPlane( ObjectMeshData& data, const DivideMeshWithPlaneParams& params )
{
    MR_TIMER;
    if ( !data.mesh )
    {
        assert( false );
        if ( params.errors )
            *params.errors = { "Invalid function input: empty mesh" };
        return;
    }

    std::function<void( EdgeId, EdgeId, float )> onEdgeSplitCallback;

    auto& mesh = *data.mesh;

    const bool updateUV = !data.uvCoordinates.empty();
    const bool updateVertColor = !data.vertColors.empty();
    const bool updateEdgeSelection = data.selectedEdges.any();
    const bool updateCreases = data.creases.any();

    UndirectedEdgeBitSet newSelectEdge;
    UndirectedEdgeBitSet otherNewSelectEdge;
    UndirectedEdgeBitSet newCreases;
    UndirectedEdgeBitSet otherNewCreases;

    if ( updateUV || updateVertColor || updateEdgeSelection || updateCreases )
    {
        onEdgeSplitCallback = [&] ( EdgeId oldEdge, EdgeId newEdge, float ratio )
        {
            const auto vo = mesh.topology.org( newEdge );
            const auto vd = mesh.topology.dest( oldEdge );
            const auto newVertId = mesh.topology.dest( newEdge );
            if ( updateUV )
                data.uvCoordinates.autoResizeSet( newVertId, lerp( data.uvCoordinates[vo], data.uvCoordinates[vd], ratio ) );

            if ( updateVertColor )
                data.vertColors.autoResizeSet( newVertId, lerp( data.vertColors[vo], data.vertColors[vd], ratio ) );

            if ( updateEdgeSelection )
            {
                if ( data.selectedEdges.test( oldEdge.undirected() ) )
                {
                    newSelectEdge.autoResizeSet( newEdge.undirected(), true );
                    if ( params.otherPart )
                        otherNewSelectEdge.autoResizeSet( oldEdge.undirected(), true );
                }
            }
            if ( updateCreases )
            {
                if ( data.creases.test( oldEdge.undirected() ) )
                {
                    newCreases.autoResizeSet( newEdge.undirected(), true );
                    if ( params.otherPart )
                        otherNewCreases.autoResizeSet( oldEdge.undirected(), true );
                }
            }
        };
    }

    const bool needUpdateFaceSelection = data.selectedFaces.any();
    FaceHashMap new2Old;
    FaceHashMap otherNew2Old;
    bool needNew2Old = !data.faceColors.empty() || needUpdateFaceSelection || !data.texturePerFace.empty();
    bool needOtherNew2Old = params.otherPart && needNew2Old;
    std::shared_ptr<Mesh> otherPartMesh;
    if ( params.otherPart )
        otherPartMesh = std::make_shared<Mesh>(); // make new mesh

    std::vector<EdgeLoop> holeContours;
    std::vector<EdgeLoop> otherHoleContours;

    trimWithPlane( mesh,
               { .plane = params.plane, .eps = params.eps, .onEdgeSplitCallback = onEdgeSplitCallback },
               { .outCutContours = params.fillCut ? &holeContours : nullptr,
               .new2Old = needNew2Old ? &new2Old : nullptr,
               .otherPart = otherPartMesh.get(),
               .otherNew2Old = needOtherNew2Old ? &otherNew2Old : nullptr,
               .otherOutCutContours = ( params.fillCut && params.otherPart ) ? &otherHoleContours : nullptr } );

    if ( params.otherPart )
    {
        *params.otherPart = data;
        params.otherPart->mesh = std::move( otherPartMesh );
    }

    auto validEdges = mesh.topology.findNotLoneUndirectedEdges();
    UndirectedEdgeBitSet otherValidEdges;
    if ( params.otherPart )
        otherValidEdges = params.otherPart->mesh->topology.findNotLoneUndirectedEdges();

    auto updateEdges = [] ( UndirectedEdgeBitSet& news, const UndirectedEdgeBitSet& olds, const UndirectedEdgeBitSet& valids )
    {
        news.resize( valids.size() );
        news |= ( valids & olds );
        news &= valids;
    };
    if ( updateEdgeSelection )
    {
        updateEdges( newSelectEdge, data.selectedEdges, validEdges );
        data.selectedEdges = std::move( newSelectEdge );
        if ( params.otherPart )
        {
            updateEdges( otherNewSelectEdge, params.otherPart->selectedEdges, otherValidEdges );
            params.otherPart->selectedEdges = std::move( otherNewSelectEdge );
        }
    }
    if ( updateCreases )
    {
        updateEdges( newCreases, data.creases, validEdges );
        data.creases = std::move( newCreases );
        if ( params.otherPart )
        {
            updateEdges( otherNewCreases, params.otherPart->creases, otherValidEdges );
            params.otherPart->creases = std::move( otherNewCreases );
        }
    }

    auto updateFaces = [] ( auto& faceAttribs, const auto& new2oldMap )
    {
        for ( auto [newId, oldId] : new2oldMap )
            faceAttribs.autoResizeSet( newId, faceAttribs[oldId] );
    };
    if ( !data.faceColors.empty() )
    {
        updateFaces( data.faceColors, new2Old );
        if ( params.otherPart )
            updateFaces( params.otherPart->faceColors, otherNew2Old );
    }
    if ( !data.texturePerFace.empty() )
    {
        updateFaces( data.texturePerFace, new2Old );
        if ( params.otherPart )
            updateFaces( params.otherPart->texturePerFace, otherNew2Old );
    }

    auto updateFaceSelection = []( FaceBitSet& selection, const FaceHashMap& new2oldMap, const FaceBitSet& valids )
    {
        selection.resize( valids.size() );
        for ( auto [newId, oldId] : new2oldMap )
            if ( selection.test( oldId ) )
                selection.set( newId );
        selection &= valids;
    };
    if ( needUpdateFaceSelection )
    {
        updateFaceSelection( data.selectedFaces, new2Old, mesh.topology.getValidFaces() );
        if ( params.otherPart )
            updateFaceSelection( params.otherPart->selectedFaces, otherNew2Old, params.otherPart->mesh->topology.getValidFaces() );
    }

    if ( !params.fillCut )
        return;

    auto fillCutContours = [subdivide = params.subdivideFilling] ( ObjectMeshData& data, std::vector<EdgeLoop>& holeContours ) -> Expected<void>
    {
        auto& mesh = *data.mesh;
        auto& tp = mesh.topology;
        float sumBdLen = 0;
        int numBdEdges = 0;
        for ( const auto& path : holeContours )
        {
            for ( auto e : path )
            {
                sumBdLen += mesh.edgeLength( e );
                ++numBdEdges;
                if ( tp.right( e ).valid() )
                    return unexpected( "Cannot cut mesh" );
            }
        }

        auto selRegion = data.selectedFaces;
        const bool someInitialSelection = selRegion.any();
        auto res = fillPlanarHole( data, holeContours ).and_then( [&] () ->Expected<void>
        {
            if ( subdivide && numBdEdges > 0 )
            {
                // subdivide big triangles on the section
                if ( someInitialSelection )
                    selRegion = data.selectedFaces - selRegion; // only faces appeared during fillPlanarHole
                const float avgBdLen = sumBdLen / numBdEdges;
                SubdivideSettings subs
                {
                    .maxEdgeLen = 2 * avgBdLen, // boundary edges are typically shorter after section than original edges, hence 2 multiplier to compensate
                    .maxDeviationAfterFlip = FLT_MAX, // section is completely planar, so we do not check deviation
                    .region = someInitialSelection ? &selRegion : nullptr // special region only if subdivide not all current selection
                };
                subdivideMesh( data, subs );
            }
            return {};
        } );
        return res;
    };

    if ( params.errors )
        params.errors->clear();

    auto fillPartRes = fillCutContours( data, holeContours );
    if ( params.errors && !fillPartRes.has_value() )
        params.errors->push_back( fillPartRes.error() );
    if ( params.otherPart )
    {
        fillPartRes = fillCutContours( *params.otherPart, otherHoleContours );
        if ( params.errors && !fillPartRes.has_value() )
            params.errors->push_back( fillPartRes.error() + " (other part)" );
    }
}

}
