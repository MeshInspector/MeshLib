#include "MRSceneColors.h"
#include <assert.h>

namespace MR
{

namespace
{
struct ColorState
{
    std::array<Color, size_t( SceneColors::Count )> colors;

    ColorState()
    {
        using namespace SceneColors;

        // color of object when it is selected
        colors[SelectedObjectMesh] = colors[SelectedObjectPoints] =
            colors[SelectedObjectLines] = colors[SelectedObjectVoxels] =
            colors[SelectedObjectDistanceMap] = Color( Vector4f{ 1.0f,0.9f,0.23f,1.0f } );
        // color if object when it is NOT selected
        colors[UnselectedObjectMesh] = colors[UnselectedObjectPoints] =
            colors[UnselectedObjectLines] = colors[UnselectedObjectVoxels] =
            colors[UnselectedObjectDistanceMap] = Color( Vector4f{ 1.0f, 1.0f, 1.0f, 1.0f } );
        // color of back faces
        colors[BackFaces] = Color( Vector4f{ 0.519f,0.625f,0.344f,1.0f } );
        // color of labels
        colors[Labels] = Color( Vector4f{ 0.0f,0.0f,0.0f,1.0f } );
        // color of edges
        colors[Edges] = Color( Vector4f{ 0.0f,0.0f,0.0f,1.0f } );
        // color of selected faces
        colors[SelectedFaces] = Color( Vector4f{ 0.8f,0.2f,0.2f,1.0f } );
        // color of selected edges
        colors[SelectedEdges] = Color( Vector4f{ 0.7f,0.2f,0.7f,1.0f } );
        // color of selected points
        colors[SelectedPoints] = Color( Vector4f{ 0.8f,0.2f,0.2f,1.0f } );

        colors[SelectedFeatures] = Color( 193, 40, 107, 255 );
        colors[UnselectedFeatures] = Color( 216, 128, 70, 255 );
        colors[FeatureBackFaces] = Color( 176, 124, 91, 255 );

        colors[SelectedFeatureDecorations] = Color( 255, 64, 192, 255 );
        colors[UnselectedFeatureDecorations] = Color( 255, 64, 192, 255 );

        colors[SelectedMeasurements] = Color( 50, 255, 240, 255 );
        colors[UnselectedMeasurements] = Color( 255, 255, 255, 255 );
    }
};

ColorState &GetColorState()
{
    static ColorState ret;
    return ret;
}
} // namespace

const Color& SceneColors::get( Type type )
{
    return GetColorState().colors[type];
}

void SceneColors::set( Type type, const Color& color )
{
    GetColorState().colors[type] = color;
}

const char* SceneColors::getName( Type type )
{
    switch ( type )
    {
        case SelectedObjectMesh:           return "SelectedObjectMesh";
        case UnselectedObjectMesh:         return "UnselectedObjectMesh";
        case SelectedObjectPoints:         return "SelectedObjectPoints";
        case UnselectedObjectPoints:       return "UnselectedObjectPoints";
        case SelectedObjectLines:          return "SelectedObjectLines";
        case UnselectedObjectLines:        return "UnselectedObjectLines";
        case SelectedObjectVoxels:         return "SelectedObjectVoxels";
        case UnselectedObjectVoxels:       return "UnselectedObjectVoxels";
        case SelectedObjectDistanceMap:    return "SelectedObjectDistanceMap";
        case UnselectedObjectDistanceMap:  return "UnselectedObjectDistanceMap";
        case BackFaces:                    return "BackFaces";
        case Labels:                       return "Labels";
        case Edges:                        return "Edges";
        case SelectedFaces:                return "SelectedFaces";
        case SelectedEdges:                return "SelectedEdges";
        case SelectedPoints:               return "SelectedPoints";
        case SelectedFeatures:             return "SelectedFeatures";
        case UnselectedFeatures:           return "UnselectedFeatures";
        case FeatureBackFaces:             return "FeatureBackFaces";
        case SelectedFeatureDecorations:   return "SelectedFeatureDecorations";
        case UnselectedFeatureDecorations: return "UnselectedFeatureDecorations";
        case SelectedMeasurements:         return "SelectedMeasurements";
        case UnselectedMeasurements:       return "UnselectedMeasurements";
        case Count:                        break;
    }
    assert( false && "Invalid enum." );
    return nullptr;
}

}
