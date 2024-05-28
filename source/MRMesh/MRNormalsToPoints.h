#pragma once

#include "MRMeshFwd.h"
#include <memory>

namespace MR
{

/// The purpose of this class is to update vertex positions given target triangle normals;
/// see the article "Static/Dynamic Filtering for Mesh Geometry"
class NormalsToPoints
{
public:
    /// builds linear system and prepares a solver for it;
    /// please call it only once for mesh, and then run as many times as you like
    /// \param guideWeight how much resulting points must be attracted to initial points, must be > 0
    MRMESH_API void prepare( const MeshTopology & topology, float guideWeight = 1 );

    /// performs one iteration consisting of projection of all triangles on planes with given normals and finding best points from them
    /// \param guide target vertex positions to avoid under-determined system
    /// \param normals target face normals
    /// \param points initial approximation on input, updated approximation on output
    /// \param maxInitialDistSq the maximum squared distance between a point and its position in (guide)
    MRMESH_API void run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points );
    MRMESH_API void run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points, float maxInitialDistSq );

    // pImpl
    class ISolver
    {
    public:
        virtual ~ISolver() = default;
        virtual void prepare( const MeshTopology & topology, float guideWeight ) = 0;
        virtual void run( const VertCoords & guide, const FaceNormals & normals, VertCoords & points, float maxInitialDistSq ) = 0;
    };
private:
    std::unique_ptr<ISolver> solver_;
};

} //namespace MR
