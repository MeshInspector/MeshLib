#pragma once

#include "MRMeshFwd.h"
#include "MRExpected.h"
#include "MRLineSegm3.h"
#include "MRId.h"

namespace MR
{

/// extracts all iso-lines from given scalar field and iso-value=0
[[nodiscard]] MRMESH_API IsoLines extractIsolines( const MeshTopology & topology,
    const VertMetric & vertValues, const FaceBitSet * region = nullptr );

/// quickly returns true if extractIsolines produce not-empty set for the same arguments
[[nodiscard]] MRMESH_API bool hasAnyIsoline( const MeshTopology & topology,
    const VertMetric & vertValues, const FaceBitSet * region = nullptr );

/// extracts all iso-lines from given scalar field and iso-value
[[nodiscard]] MRMESH_API IsoLines extractIsolines( const MeshTopology & topology,
    const VertScalars & vertValues, float isoValue, const FaceBitSet * region = nullptr );

/// quickly returns true if extractIsolines produce not-empty set for the same arguments
[[nodiscard]] MRMESH_API bool hasAnyIsoline( const MeshTopology & topology,
    const VertScalars & vertValues, float isoValue, const FaceBitSet * region = nullptr );

/// extracts all plane sections of given mesh
[[nodiscard]] MRMESH_API PlaneSections extractPlaneSections( const MeshPart & mp, const Plane3f & plane );

/// quickly returns true if extractPlaneSections produce not-empty set for the same arguments
[[nodiscard]] MRMESH_API bool hasAnyPlaneSection( const MeshPart & mp, const Plane3f & plane );

/// extracts all sections of given mesh with the plane z=zLevel;
/// this function works faster than general extractPlaneSections(...) for the same plane
/// if the sections cross relatively small number of mesh triangles and AABB tree has already been constructed
[[nodiscard]] MRMESH_API PlaneSections extractXYPlaneSections( const MeshPart & mp, float zLevel );

/// quickly returns true if extractXYPlaneSections produce not-empty set for the same arguments
[[nodiscard]] MRMESH_API bool hasAnyXYPlaneSection( const MeshPart & mp, float zLevel );

struct TriangleSection
{
    LineSegm3f segm;
    FaceId f;
};
[[nodiscard]] MRMESH_API std::vector<TriangleSection> findTriangleSectionsByXYPlane( const MeshPart & mp, float zLevel );

/// track section of plane set by start point, direction and surface normal in start point 
/// in given direction while given distance or
/// mesh boundary is not reached, or track looped
/// negative distance means moving in opposite direction
/// returns track on surface and end point (same as start if path has looped)
[[nodiscard]] MRMESH_API PlaneSection trackSection( const MeshPart& mp,
    const MeshTriPoint& start, MeshTriPoint& end, const Vector3f& direction, float distance );

/// track section of plane set by start point, end point and planePoint
/// from start to end
/// \param ccw - if true use start->end->planePoint plane, otherwise use start->planePoint->end (changes direction of plane tracking)
/// returns track on surface without end point (return error if path was looped or reached boundary)
[[nodiscard]] MRMESH_API Expected<PlaneSection> trackSection( const MeshPart& mp,
    const MeshTriPoint& start, const MeshTriPoint& end, const Vector3f& planePoint, bool ccw );

/// returns true if left(isoline[i].e) == right(isoline[i+1].e) and valid for all i;
/// all above functions produce consistently oriented lines
[[nodiscard]] MRMESH_API bool isConsistentlyOriented( const MeshTopology & topology, const IsoLine & isoline );

/// for a consistently oriented isoline, returns all faces it goes inside
[[nodiscard]] MRMESH_API FaceBitSet getCrossedFaces( const MeshTopology & topology, const IsoLine & isoline );

/// converts PlaneSections in 2D contours by computing coordinate of each point, applying given xf to it, and retaining only x and y
[[nodiscard]] MRMESH_API Contour2f planeSectionToContour2f( const Mesh & mesh, const PlaneSection & section, const AffineXf3f & meshToPlane );

[[nodiscard]] MRMESH_API Contours2f planeSectionsToContours2f( const Mesh & mesh, const PlaneSections & sections, const AffineXf3f & meshToPlane );

} //namespace MR
