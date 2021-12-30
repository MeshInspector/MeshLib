#include "MRMeshColors.h"
#include <assert.h>

namespace MR
{
MeshColors& MeshColors::instance_()
{
    static MeshColors instance_;
    return instance_;
}

const Color& MeshColors::get( Type type )
{
    return instance_().colors_[type];
}

void MeshColors::set( Type type, const Color& color )
{
    instance_().colors_[type] = color;
}

const char* MeshColors::getName( Type type )
{
    switch ( type )
    {
    case MR::MeshColors::SelectedMesh: return "SelectedMesh";
    case MR::MeshColors::UnselectedMesh: return "UnselectedMesh";
    case MR::MeshColors::BackFaces: return "BackFaces";
    case MR::MeshColors::Labels: return "Labels";
    case MR::MeshColors::Edges: return "Edges";
    case MR::MeshColors::SelectedFaces: return "SelectedFaces";
    case MR::MeshColors::SelectedEdges: return "SelectedEdges";
    default:
        assert( false );
    }
    return "";
}

MeshColors::MeshColors()
{
    // color of object when it is selected
    colors_[SelectedMesh] = Color( Vector4f{ 1.0f,0.9f,0.23f,1.0f } );
    // color if object when it is NOT selected
    colors_[UnselectedMesh] = Color( Vector4f{ 1.0f, 1.0f, 1.0f, 1.0f } );
    // color of back faces
    colors_[BackFaces] = Color( Vector4f{ 0.519f,0.625f,0.344f,1.0f } );
    // color of labels
    colors_[Labels] = Color( Vector4f{ 0.0f,0.0f,0.0f,1.0f } );
    // color of edges
    colors_[Edges] = Color( Vector4f{ 0.0f,0.0f,0.0f,1.0f } );
    // color of selected faces
    colors_[SelectedFaces] = Color( Vector4f{ 0.8f,0.2f,0.2f,1.0f } );
    // color of selected edges
    colors_[SelectedEdges] = Color( Vector4f{ 0.7f,0.2f,0.7f,1.0f } );
}

}
