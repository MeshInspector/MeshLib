#pragma once
#include "MRMeshFwd.h"
#include "MRColor.h"
#include "MRVector4.h"

#include <array>

namespace MR
{

/// Contains default colors for scene objects
/// \ingroup BasicStructuresGroup
namespace SceneColors
{
    enum Type
    {
        SelectedObjectMesh,
        UnselectedObjectMesh,
        SelectedObjectPoints,
        UnselectedObjectPoints,
        SelectedObjectLines,
        UnselectedObjectLines,
        SelectedObjectVoxels,
        UnselectedObjectVoxels,
        SelectedObjectDistanceMap,
        UnselectedObjectDistanceMap,
        BackFaces,
        Labels,
        Edges,
        SelectedFaces,
        SelectedEdges,
        SelectedPoints,
        SelectedFeatures,
        UnselectedFeatures,
        FeatureBackFaces,
        SelectedFeatureDecorations,
        UnselectedFeatureDecorations,
        SelectedMeasurements,
        UnselectedMeasurements,
        Count [[maybe_unused]],
    };

    MRMESH_API const Color& get( Type type );
    MRMESH_API void set( Type type, const Color& color );

    MRMESH_API const char* getName( Type type );
}

}
