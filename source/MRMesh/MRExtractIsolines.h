#pragma once

#include "MRMeshFwd.h"

namespace MR
{

using IsoLine = std::vector<MeshEdgePoint>;
using IsoLines = std::vector<IsoLine>;

using PlaneSection = std::vector<MeshEdgePoint>;
using PlaneSections = std::vector<PlaneSection>;

// extracts all iso-lines from given scalar field and iso-value
MRMESH_API IsoLines extractIsolines( const MeshTopology & topology,
    const Vector<float,VertId> & vertValues, float isoValue, const FaceBitSet * region = nullptr );

// extracts all plane sections of given mesh
MRMESH_API PlaneSections extractPlaneSections( const MeshPart & mp, const Plane3f & plane );

// converts PlaneSections in 2D contours by computing coordinate of each point, applying given xf to it, and retaining only x and y
MRMESH_API Contour2f planeSectionToContour2f( const Mesh & mesh, const PlaneSection & section, const AffineXf3f & meshToPlane );
MRMESH_API Contours2f planeSectionsToContours2f( const Mesh & mesh, const PlaneSections & sections, const AffineXf3f & meshToPlane );

} //namespace MR
