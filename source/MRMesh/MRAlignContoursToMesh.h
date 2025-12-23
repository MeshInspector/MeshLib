#pragma once
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRId.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRMeshTriPoint.h"

namespace MR
{

/// Parameters for aligning 2d contours onto mesh surface
struct ContoursMeshAlignParams
{
    /// Point coordinate on mesh, represent position of contours box 'pivotPoint' on mesh
    MeshTriPoint meshPoint;
    
    /// Relative position of 'meshPoint' in contours bounding box
    /// (0, 0) - bottom left, (0, 1) - bottom right, (0.5, 0.5) - center, (1, 1) - top right
    Vector2f pivotPoint{ 0.0f, 0.0f };
    
    /// Represents 2d contours xDirection in mesh space
    Vector3f xDirection;

    /// Represents contours normal in mesh space 
    /// if nullptr - use mesh normal at 'meshPoint'
    const Vector3f* zDirection{ nullptr };

    /// Contours extrusion in +z and -z direction
    float extrusion{ 1.0f };

    /// Maximum allowed shift along 'zDirection' for alignment
    float maximumShift{ 2.5f };
};

/// Creates planar mesh out of given contour and aligns it to given surface
MRMESH_API Expected<Mesh> alignContoursToMesh( const Mesh& mesh, const Contours2f& contours, const ContoursMeshAlignParams& params );

/// Parameters for aligning 2d contours onto mesh surface along given curve
struct ContoursMeshCurvedAlignParams
{
    /// Relative position of curve line (y=pivotY) in contours bounding box:
    /// 0 - bottom, 0.5 - center, 1 - top
    float pivotY = 0;

    /// converts (x, pivotY) into position on curve, which must be close enough to the mesh
    std::function<Vector3f(float x)> curvePos;

    /// converts (x, pivotY) into unit direction along curve
    std::function<Vector3f(float x)> curveDir;

    /// Contours extrusion outside of mesh
    float extrusion{ 1.0f };
};

/// Converts contours in thick mesh, and deforms it to be properly aligned on the surface of given mesh
MRMESH_API Expected<Mesh> curvedAlignContoursToMesh( const Mesh& mesh, const Contours2f& contours, const ContoursMeshCurvedAlignParams& params );

/// given a planar mesh with boundary on input located in plane XY, packs and extends it along Z on zOffset (along -Z if zOffset is negative) to make a volumetric closed mesh
/// note that this function also packs the mesh
MRMESH_API void addBaseToPlanarMesh( Mesh& mesh, float zOffset );

}