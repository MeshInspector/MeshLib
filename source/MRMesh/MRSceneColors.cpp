#include "MRSceneColors.h"
#include <assert.h>

namespace MR
{
SceneColors& SceneColors::instance_()
{
    static SceneColors instance_;
    return instance_;
}

const Color& SceneColors::get( Type type )
{
    return instance_().colors_[type];
}

void SceneColors::set( Type type, const Color& color )
{
    instance_().colors_[type] = color;
}

const char* SceneColors::getName( Type type )
{
    constexpr std::array<const char*, size_t( Type::Count )> names =
    {
        "SelectedObjectMesh",
        "UnselectedObjectMesh",
        "SelectedObjectPoints",
        "UnselectedObjectPoints",
        "SelectedObjectLines",
        "UnselectedObjectLines",
        "SelectedObjectVoxels",
        "UnselectedObjectVoxels",
        "SelectedObjectDistanceMap",
        "UnselectedObjectDistanceMap",
        "BackFaces",
        "Labels",
        "Edges",
        "SelectedFaces",
        "SelectedEdges",
        "SelectedPoints"
    };
    return names[int( type )];
}

SceneColors::SceneColors()
{
    // color of object when it is selected
    colors_[SelectedObjectMesh] = colors_[SelectedObjectPoints] = 
        colors_[SelectedObjectLines] = colors_[SelectedObjectVoxels] = 
        colors_[SelectedObjectDistanceMap] = Color( Vector4f{ 1.0f,0.9f,0.23f,1.0f } );
    // color if object when it is NOT selected
    colors_[UnselectedObjectMesh] = colors_[UnselectedObjectPoints] = 
        colors_[UnselectedObjectLines] = colors_[UnselectedObjectVoxels] = 
        colors_[UnselectedObjectDistanceMap] = Color( Vector4f{ 1.0f, 1.0f, 1.0f, 1.0f } );
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
    // color of selected points
    colors_[SelectedPoints] = Color( Vector4f{ 0.8f,0.2f,0.2f,1.0f } );
}

}
