#pragma once
#include "MRMeshFwd.h"
#include "MRColor.h"
#include "MRVector4.h"

#include <array>

namespace MR
{

// This singleton struct contain default colors for meshes
struct MeshColors
{
    enum Type
    {
        SelectedMesh,
        UnselectedMesh,
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
    MeshColors();
    MeshColors( const MeshColors& ) = delete;
    MeshColors( MeshColors&& ) = delete;
    ~MeshColors() = default;

    static MeshColors& instance_();

    std::array<Color, size_t( Count )> colors_;
};

}