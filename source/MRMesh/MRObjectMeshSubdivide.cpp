#include "MRObjectMeshSubdivide.h"
#include "MRMeshSubdivide.h"
#include "MRMeshAttributesToUpdate.h"
#include "MRMeshSubdivideCallbacks.h"
#include "MRObjectMesh.h"
#include "MRTimer.h"

namespace MR
{

ObjectMeshSubdivideResult subdivideObjectMesh( const ObjectMesh& obj, const SubdivideSettings& subs )
{
    MR_TIMER

    ObjectMeshSubdivideResult res;
    if ( !obj.mesh() )
    {
        assert( false );
        return res;
    }

    res.selEdges = obj.getSelectedEdges();
    res.creases = obj.creases();
    auto notFlippable = res.selEdges | res.creases;
    res.uvCoords = obj.getUVCoords();
    res.colorMap = obj.getVertsColorMap();

    res.texturePerFace = obj.getTexturePerFace();
    res.faceColors = obj.getFacesColorMap();

    MeshAttributesToUpdate meshParams;
    if ( !res.uvCoords.empty() )
        meshParams.uvCoords = &res.uvCoords;
    if ( !res.colorMap.empty() )
        meshParams.colorMap = &res.colorMap;
    if ( !res.texturePerFace.empty() )
        meshParams.texturePerFace = &res.texturePerFace;
    if ( !res.faceColors.empty() )
        meshParams.faceColors = &res.faceColors;

    res.mesh = *obj.mesh();
    auto updateAttributesCb = meshOnEdgeSplitAttribute( res.mesh, meshParams );

    auto subs1 = subs;
    subs1.onEdgeSplit = [&] ( EdgeId e1, EdgeId e )
    {
        if ( res.selEdges.test( e.undirected() ) )
        {
            res.selEdges.autoResizeSet( e1.undirected() );
            notFlippable.autoResizeSet( e1.undirected() );
        }
        if ( res.creases.test( e.undirected() ) )
        {
            res.creases.autoResizeSet( e1.undirected() );
            notFlippable.autoResizeSet( e1.undirected() );
        }

        updateAttributesCb( e1, e );
        if ( subs.onEdgeSplit )
            subs.onEdgeSplit( e1, e );
    };
    assert( !subs1.notFlippable );
    subs1.notFlippable = &notFlippable;

    assert( !subs1.region );
    res.selFaces = obj.getSelectedFaces();
    if ( res.selFaces.any() )
        subs1.region = &res.selFaces;

    subdivideMesh( res.mesh, subs1 );
    return res;
}

void ObjectMeshSubdivideResult::assingNoHistory( ObjectMesh & target )
{
    MR_TIMER

    target.updateMesh( std::make_shared<Mesh>( std::move( mesh ) ) );
    target.setUVCoords( std::move( uvCoords ) );
    target.setVertsColorMap( std::move( colorMap ) );
    target.setTexturePerFace( std::move( texturePerFace ) );
    target.setFacesColorMap( std::move( faceColors ) );
    target.selectFaces( std::move( selFaces ) );
    target.selectEdges( std::move( selEdges ) );
    target.setCreases( std::move( creases ) );
}

} //namespace MR
