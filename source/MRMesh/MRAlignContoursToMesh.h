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

    /// if true, curve parameter will be always within [0,1) with repetition: xr := x - floor(x)
    bool periodicCurve = false;

    /// stretch all contours along curve to fit in unit curve range
    bool stretch = true;

    /// Contours extrusion outside of curve level
    float extrusion{ 1.0f };

    /// To allow passing Python lambdas into `curve`.
    MR_BIND_PREFER_UNLOCK_GIL_WHEN_USED_AS_PARAM
};

/// Converts contours in thick mesh, and deforms it along given path
MRMESH_API Expected<Mesh> bendContoursAlongCurve( const Contours2f& contours, const CurveFunc& curve, const BendContoursAlongCurveParams& params );

/// Converts contours in thick mesh, and deforms it along given surface path: start->path->end
MRMESH_API Expected<Mesh> bendContoursAlongSurfacePath( const Contours2f& contours, const Mesh& mesh, const MeshTriPoint & start, const SurfacePath& path, const MeshTriPoint & end,
    const BendContoursAlongCurveParams& params );

/// Converts contours in thick mesh, and deforms it along given surface path
MRMESH_API Expected<Mesh> bendContoursAlongSurfacePath( const Contours2f& contours, const Mesh& mesh, const SurfacePath& path,
    const BendContoursAlongCurveParams& params );

/// given a polyline by its vertices, computes partial lengths along the polyline from the initial point;
/// return an error if the polyline is less than 2 points or all points have exactly the same location
/// \param unitLength if true, then the lengths are normalized for the last point to have unit length
/// \param outCurveLen optional output of the total polyline length (before possible normalization)
MRMESH_API Expected<std::vector<float>> findPartialLens( const CurvePoints& cp, bool unitLength = true, float * outCurveLen = nullptr );

/// given a polyline by its vertices, and partial lengths as computed by \ref findPartialLens,
/// finds the location of curve point at the given parameter with extrapolation if p outside [0, lens.back()],
/// execution time is logarithmic relative to the number of points
[[nodiscard]] MRMESH_API CurvePoint getCurvePoint( const CurvePoints& cp, const std::vector<float> & lens, float p );

/// given a polyline by its vertices, returns curve function representing it;
/// return an error if the polyline is less than 2 points or all points have exactly the same location
/// \param unitLength if true, then the lengths are normalized for the last point to have unit length
/// \param outCurveLen optional output of the total polyline length (before possible normalization)
MRMESH_API Expected<CurveFunc> curveFromPoints( const CurvePoints& cp, bool unitLength = true, float * outCurveLen = nullptr );
MRMESH_API Expected<CurveFunc> curveFromPoints( CurvePoints&& cp, bool unitLength = true, float * outCurveLen = nullptr );

/// converts polyline given as a number of MeshTriPoint/MeshEdgePoint into CurvePoints
[[nodiscard]] MRMESH_API CurvePoints meshPathCurvePoints( const Mesh& mesh, const MeshTriPoint & start, const SurfacePath& path, const MeshTriPoint & end );
[[nodiscard]] MRMESH_API CurvePoints meshPathCurvePoints( const Mesh& mesh, const SurfacePath& path );

/// given a planar mesh with boundary on input located in plane XY, packs and extends it along Z on zOffset (along -Z if zOffset is negative) to make a volumetric closed mesh
/// note that this function also packs the mesh
MRMESH_API void addBaseToPlanarMesh( Mesh& mesh, float zOffset );

}