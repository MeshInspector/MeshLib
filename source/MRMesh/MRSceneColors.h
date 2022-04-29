#pragma once
#include "MRMeshFwd.h"
#include "MRColor.h"
#include "MRVector4.h"

#include <array>

namespace MR
{

/// This singleton struct contains default colors for scene objects
/// \ingroup BasicStructuresGroup
struct SceneColors
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
        Count
    };

    MRMESH_API static const Color& get( Type type );
    MRMESH_API static void set( Type type, const Color& color );

    MRMESH_API static const char* getName( Type type );

private:
    SceneColors();
    SceneColors( const SceneColors& ) = delete;
    SceneColors( SceneColors&& ) = delete;
    ~SceneColors() = default;

    static SceneColors& instance_();

    std::array<Color, size_t( Count )> colors_;
};

}