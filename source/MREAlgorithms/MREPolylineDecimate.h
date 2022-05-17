#pragma once
#include "exports.h"
#include "MRMesh/MRMeshFwd.h"
#include <climits>
#include <functional>

namespace MRE
{

/**
 * \struct MRE::DecimatePolylineSettings
 * \brief Parameters structure for MRE::decimatePolyline
 * \ingroup DecimateGroup
 *
 * \sa \ref decimatePolyline
 */
template<typename V>
struct DecimatePolylineSettings
{
    /// Limit from above on the maximum distance from moved vertices to original contour
    float maxError = 0.001f;
    /// Small stabilizer is important to achieve good results on completely linear polyline parts,
    /// if your polyline is not-linear everywhere, then you can set it to zero
    float stabilizer = 0.001f;
    /// Limit on the number of deleted vertices
    int maxDeletedVertices = INT_MAX;
    /// Region of the polyline to be decimated, it is updated during the operation
    /// Remain nullptr to include the whole polyline
    MR::VertBitSet* region = nullptr;
    /// Whether to allow collapsing edges with at least one vertex on (region) boundary
    bool touchBdVertices = true;
    /**
     * \brief The user can provide this optional callback that is invoked immediately before edge collapse;
     * \details It receives the edge being collapsed: its destination vertex will disappear,
     * and its origin vertex will get new position (provided as the second argument) after collapse;
     * If the callback returns false, then the collapse is prohibited
     */
    std::function<bool( MR::EdgeId edgeToCollapse, const V & newEdgeOrgPos )> preCollapse;
    /**
     * \brief  If not null, then
     * on input: if the vector is not empty then it is takes for initialization instead of form computation for all vertices;
     * on output: quadratic form for each remaining vertex is returned there
     */
    MR::Vector<MR::QuadraticForm<V>, MR::VertId>* vertForms = nullptr;
};

using DecimatePolylineSettings2 = DecimatePolylineSettings<MR::Vector2f>;
using DecimatePolylineSettings3 = DecimatePolylineSettings<MR::Vector3f>;

/**
 * \struct MRE::DecimatePolylineResult
 * \brief Results of MRE::decimateContour
 */
struct DecimatePolylineResult
{
    int vertsDeleted = 0; ///< Number deleted verts. Same as the number of performed collapses
    float errorIntroduced = 0; ///< Max different (as distance) between original contour and result contour
};

/**
 * \brief Collapse edges in the polyline according to the settings
 * \ingroup DecimateGroup
 */
MREALGORITHMS_API DecimatePolylineResult decimatePolyline( MR::Polyline2& polyline, const DecimatePolylineSettings2& settings = {} );
MREALGORITHMS_API DecimatePolylineResult decimatePolyline( MR::Polyline3& polyline, const DecimatePolylineSettings3& settings = {} );

/**
 * \brief Collapse edges in the contour according to the settings
 * \ingroup DecimateGroup
 */ 
MREALGORITHMS_API DecimatePolylineResult decimateContour( MR::Contour2f& contour, const DecimatePolylineSettings2& settings = {} );
MREALGORITHMS_API DecimatePolylineResult decimateContour( MR::Contour3f& contour, const DecimatePolylineSettings3& settings = {} );

} //namespace MRE
