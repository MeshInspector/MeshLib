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
    switch ( type )
    {
        case SelectedObjectMesh:              return "SelectedObjectMesh";
        case UnselectedObjectMesh:            return "UnselectedObjectMesh";
        case SelectedObjectPoints:            return "SelectedObjectPoints";
        case UnselectedObjectPoints:          return "UnselectedObjectPoints";
        case SelectedObjectLines:             return "SelectedObjectLines";
        case UnselectedObjectLines:           return "UnselectedObjectLines";
        case SelectedObjectVoxels:            return "SelectedObjectVoxels";
        case UnselectedObjectVoxels:          return "UnselectedObjectVoxels";
        case SelectedObjectDistanceMap:       return "SelectedObjectDistanceMap";
        case UnselectedObjectDistanceMap:     return "UnselectedObjectDistanceMap";
        case BackFaces:                       return "BackFaces";
        case Labels:                          return "Labels";
        case LabelsGood:                      return "LabelsGood";
        case LabelsBad:                       return "LabelsBad";
        case Edges:                           return "Edges";
        case Points:                          return "Points";
        case SelectedFaces:                   return "SelectedFaces";
        case SelectedEdges:                   return "SelectedEdges";
        case SelectedPoints:                  return "SelectedPoints";
        case SelectedFeatures:                return "SelectedFeatures";
        case UnselectedFeatures:              return "UnselectedFeatures";
        case FeatureBackFaces:                return "FeatureBackFaces";
        case SelectedFeatureDecorations:      return "SelectedFeatureDecorations";
        case UnselectedFeatureDecorations:    return "UnselectedFeatureDecorations";
        case SelectedMeasurements:            return "SelectedMeasurements";
        case UnselectedMeasurements:          return "UnselectedMeasurements";
        case SelectedTemporaryMeasurements:   return "SelectedTemporaryMeasurements";
        case UnselectedTemporaryMeasurements: return "UnselectedTemporaryMeasurements";
        case Count:                           break;
    }
    assert( false && "Invalid enum." );
    return nullptr;
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
    // color of vertices
    colors_[Points] = Color( Vector4f{ 0.282f, 0.965f, 0.0f, 1.0f } );
    // color of selected faces
    colors_[SelectedFaces] = Color( Vector4f{ 0.8f,0.2f,0.2f,1.0f } );
    // color of selected edges
    colors_[SelectedEdges] = Color( Vector4f{ 0.7f,0.2f,0.7f,1.0f } );
    // color of selected points
    colors_[SelectedPoints] = Color( Vector4f{ 0.8f,0.2f,0.2f,1.0f } );

    colors_[SelectedFeatures] = Color( 193, 40, 107, 255 );
    colors_[UnselectedFeatures] = Color( 216, 128, 70, 255 );
    colors_[FeatureBackFaces] = Color( 176, 124, 91, 255 );

    colors_[SelectedFeatureDecorations] = Color( 255, 64, 192, 255 );
    colors_[UnselectedFeatureDecorations] = Color( 255, 64, 192, 255 );

    colors_[SelectedMeasurements] = Color( 50, 255, 240, 255 );
    colors_[UnselectedMeasurements] = Color( 255, 255, 255, 255 );
}

}
