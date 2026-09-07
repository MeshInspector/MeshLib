#pragma once

#include "MRHoleFillPlan.h"
#include "MRExpected.h"
#include <memory>

namespace MR
{

/**
 * @brief fill holes with border in same plane (i.e. after cut by plane)
 * @param mesh - mesh with holes
 * @param holeRepresentativeEdges - each edge here represents a hole borders that should be filled
 * should be not empty
 * edges should have invalid left face (FaceId == -1)
 * @return Expected with has_value()=true if holes filled, otherwise - string error
 */
MRMESH_API Expected<void> fillContours2D( Mesh& mesh, const std::vector<EdgeId>& holeRepresentativeEdges );

/// keeps the buffers of \ref fillContours2DPlan alive between calls (the sweep-line triangulation's
/// own cache among them), so a caller preparing plans for many holes one by one avoids re-allocating
/// them on every call; one cache must not be used by several threads at once
class IFillContours2DPlanCache
{
public:
    IFillContours2DPlanCache() = default;
    /// not copyable: the cache is its owner's private scratch, and a copy would silently duplicate
    /// every buffer (this also silences warning C5267 about the user-provided destructor below)
    IFillContours2DPlanCache( const IFillContours2DPlanCache & ) = delete;
    IFillContours2DPlanCache & operator =( const IFillContours2DPlanCache & ) = delete;
    /// pure to make the class abstract: instances are created by makeFillContours2DPlanCache() only
    MRMESH_API virtual ~IFillContours2DPlanCache() = 0;
};

/// creates a cache for \ref fillContours2DPlan
MRMESH_API std::unique_ptr<IFillContours2DPlanCache> makeFillContours2DPlanCache();

/**
 * @brief prepare filling plan for hole with border in same plane (i.e. after cut by plane)
 * @param mesh - mesh with hole
 * @param holeEdgeId - the edge here represents a hole borders that should be filled
 * edge should have invalid left face (FaceId == -1)
 * @param cache - if not null, keeps the buffers of this call in it, see IFillContours2DPlanCache
 * @return Expected with has_value()=true if hole plan is prepared, otherwise - string error
 */
MRMESH_API Expected<HoleFillPlan> fillContours2DPlan( const Mesh& mesh, EdgeId holeEdgeId, IFillContours2DPlanCache* cache = nullptr );

/// computes the transformation that maps
/// O into center mass of contours' points
/// OXY into best plane containing the points
MRMESH_API AffineXf3f getXfFromOxyPlane( const Contours3f& contours );
MRMESH_API AffineXf3f getXfFromOxyPlane( const Mesh& mesh, const std::vector<EdgePath>& paths );

/// given an ObjectMeshData and the contours of a planar hole in it,
/// fills the hole using fillContours2D function and updates all per-element attributes;
/// if some contours were not closed on input, then it closes them by adding a bridge edge in each
MRMESH_API Expected<void> fillPlanarHole( ObjectMeshData& data, std::vector<EdgeLoop>& holeContours );

} //namespace MR
