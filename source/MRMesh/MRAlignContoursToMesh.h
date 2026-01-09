#pragma once
#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRId.h"
#include "MRVector2.h"
#include "MRVector3.h"
#include "MRMeshTriPoint.h"
#include "MRCurve.h"

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

/// Parameters for aligning 2d contours along given curve
struct BendContoursAlongCurveParams
{
    /// Position on the curve, where bounding box's pivot point is mapped
    float pivotCurveTime = 0;

    /// Position of the curve(pivotCurveTime) in the contours' bounding box:
    /// (0, 0) - bottom left, (0, 1) - bottom right, (0.5, 0.5) - center, (1, 1) - top right
    Vector2f pivotBoxPoint{0.0f, 0.0f};

    /// converts (x in [0,1], pivotY) into position on curve
    CurveFunc curve;

    /// if true, curve parameter will be always within [0,1) with repetition: xr := x - floor(x)
    bool periodicCurve = false;

    /// stretch all contours along curve to fit in unit curve range
    bool stretch = true;

    /// Contours extrusion outside of curve level
    float extrusion{ 1.0f };
};

/// Converts contours in thick mesh, and deforms it along given path
MRMESH_API Expected<Mesh> bendContoursAlongCurve( const Contours2f& contours, const BendContoursAlongCurveParams& params );

/// given a planar mesh with boundary on input located in plane XY, packs and extends it along Z on zOffset (along -Z if zOffset is negative) to make a volumetric closed mesh
/// note that this function also packs the mesh
MRMESH_API void addBaseToPlanarMesh( Mesh& mesh, float zOffset );

}