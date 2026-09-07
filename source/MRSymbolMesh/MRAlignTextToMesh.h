#pragma once

#include "MRSymbolMeshFwd.h"
#include "MRSymbolMesh.h"

#include "MRMesh/MRExpected.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRMeshTriPoint.h"
#include "MRMesh/MRCurve.h"

namespace MR
{

struct TextMeshAlignParams : SymbolMeshParams
{
    /// Position on mesh, where text's pivot point is mapped
    MeshTriPoint startPoint;

    /// Position of the startPoint in a text bounding box
    /// (0, 0) - bottom left, (0, 1) - bottom right, (0.5, 0.5) - center, (1, 1) - top right
    Vector2f pivotPoint{0.0f, 0.0f};

    /// Direction of text, must be not zero
    Vector3f direction;

    /// Text normal to surface, if nullptr - use mesh normal at `startPoint`
    const Vector3f* textNormal{nullptr};

    /// Font height, meters
    float fontHeight{1.0f};

    /// Text mesh inside and outside offset of input mesh
    float surfaceOffset{1.0f};

    /// Maximum possible movement of text mesh alignment, meters
    float textMaximumMovement{2.5f};

    /// If true then size of each symbol will be calculated from font height, otherwise - on bounding box of the text
    bool fontBasedSizeCalc{ false };
};

/// Creates symbol mesh and aligns it to given surface
MRSYMBOLMESH_API Expected<Mesh> alignTextToMesh( const Mesh& mesh, const TextMeshAlignParams& params );

struct BendTextAlongCurveParams
{
    /// parameters of contours to mesh conversion
    SymbolMeshParams symbolMesh;

    /// Position on the curve, where bounding box's pivot point is mapped
    float pivotCurveTime = 0;

    /// Position of the curve(pivotCurveTime) in the test' bounding box:
    /// (0, 0) - bottom left, (0, 1) - bottom right, (0.5, 0.5) - center, (1, 1) - top right
    Vector2f pivotBoxPoint{0.0f, 0.0f};

    // Font height, meters
    float fontHeight{1.0f};

    // Text mesh inside and outside offset of curve's surface
    float surfaceOffset{1.0f};

    // If true then size of each symbol will be calculated from font height, otherwise - on bounding box of the text
    bool fontBasedSizeCalc{ false };

    /// if true, curve parameter will be always within [0,1) with repetition: xr := x - floor(x)
    bool periodicCurve = false;

    /// stretch whole text along curve to fit in unit curve range
    bool stretch = true;
};

/// Creates symbol mesh and deforms it along given curve
/// \param curve converts (x in [0,1], pivotY) into position on curve
MRSYMBOLMESH_API Expected<Mesh> bendTextAlongCurve( const CurveFunc& curve, const BendTextAlongCurveParams& params );

/// Creates symbol mesh and deforms it along given curve
MRSYMBOLMESH_API Expected<Mesh> bendTextAlongCurve( const CurvePoints& curve, const BendTextAlongCurveParams& params );

/// Creates symbol mesh and deforms it along given surface path: start->path->end
MRSYMBOLMESH_API Expected<Mesh> bendTextAlongSurfacePath( const Mesh& mesh,
    const GeodesicPath& path, const BendTextAlongCurveParams& params );

/// Creates symbol mesh and deforms it along given surface path
MRSYMBOLMESH_API Expected<Mesh> bendTextAlongSurfacePath( const Mesh& mesh,
    const SurfacePath& path, const BendTextAlongCurveParams& params );

} // namespace MR
