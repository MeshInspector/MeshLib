#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// extracts all iso-lines from given scalar field and iso-value=0
MRMESH_API IsoLines extractIsolines( const MeshTopology & topology,
    const VertToFloatFunc & vertValues, const FaceBitSet * region = nullptr );
/// quickly returns true if extractIsolines produce not-empty set for the same arguments
MRMESH_API bool hasAnyIsoline( const MeshTopology & topology,
    const VertToFloatFunc & vertValues, const FaceBitSet * region = nullptr );

/// extracts all iso-lines from given scalar field and iso-value
MRMESH_API IsoLines extractIsolines( const MeshTopology & topology,
    const VertScalars & vertValues, float isoValue, const FaceBitSet * region = nullptr );
/// quickly returns true if extractIsolines produce not-empty set for the same arguments
MRMESH_API bool hasAnyIsoline( const MeshTopology & topology,
    const VertScalars & vertValues, float isoValue, const FaceBitSet * region = nullptr );

/// extracts all plane sections of given mesh
MRMESH_API PlaneSections extractPlaneSections( const MeshPart & mp, const Plane3f & plane );
/// quickly returns true if extractPlaneSections produce not-empty set for the same arguments
MRMESH_API bool hasAnyPlaneSection( const MeshPart & mp, const Plane3f & plane );

/// track section of plane set by start point, direction and surface normal in start point 
/// in given direction while given distance or
/// mesh boundary is not reached, or track looped
/// negative distance means moving in opposite direction
/// returns track on surface and end point (same as start if path has looped)
MRMESH_API PlaneSection trackSection( const MeshPart& mp,
    const MeshTriPoint& start, MeshTriPoint& end, const Vector3f& direction, float distance );

/// converts PlaneSections in 2D contours by computing coordinate of each point, applying given xf to it, and retaining only x and y
MRMESH_API Contour2f planeSectionToContour2f( const Mesh & mesh, const PlaneSection & section, const AffineXf3f & meshToPlane );
MRMESH_API Contours2f planeSectionsToContours2f( const Mesh & mesh, const PlaneSections & sections, const AffineXf3f & meshToPlane );

} //namespace MR
