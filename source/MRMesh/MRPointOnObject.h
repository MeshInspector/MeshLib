#pragma once

#include "MRPch/MRBindingMacros.h"
#include "MRPointOnFace.h"
#include "MRMeshTriPoint.h"
#include "MREdgePoint.h"
#include <optional>
#include <variant>

namespace MR
{

// point located on either
// 1. face of ObjectMesh
// 2. line of ObjectLines
// 3. point of ObjectPoints
struct PointOnObject
{
    PointOnObject() {} //default ctor is required by Clang
    /// 3D location on the object in local coordinates
    Vector3f point;
    /// z buffer value
    float zBuffer{ 1.0f };
    /// to which primitive that point pertains
    union
    {
        int primId = -1;
        FaceId face;             //for ObjectMesh
        UndirectedEdgeId uedge;  //for ObjectLines
        VertId vert;             //for ObjectPoints
    };
    [[nodiscard]] operator PointOnFace() const { return { .face = face, .point = point }; }
};

// For long-term storage of picked points on an object, such as point and contour widgets,
// it is more convenient to use the local coordinates of the object rather than 3D coordinates,
// which can change depending on the xf of the object.
// --- MeshTriPoint for ObjectMeshHolder
// --- EdgePoint for ObjectLinesHolder (polylines)
// --- VertId for ObjectPointsHolder
// --- std::monostate means not valid pick (pick in empty space).
using PickedPoint = std::variant<std::monostate, MeshTriPoint, EdgePoint, VertId>;

/// Converts PointOnObject coordinates depending on the object type to the PickedPoint variant
MRMESH_API PickedPoint pointOnObjectToPickedPoint( const VisualObject* object, const PointOnObject& pos );

/// Converts given point into local coordinates of its object,
/// returns std::nullopt if object or point is invalid, or if it does not present in the object's topology
MRMESH_API std::optional<Vector3f> getPickedPointPosition( const VisualObject& object, const PickedPoint& point );

/// Converts pickedPoint into local coordinates of its object
[[deprecated( "use getPickedPointPosition() instead" )]] MRMESH_API MR_BIND_IGNORE Vector3f pickedPointToVector3( const VisualObject* object, const PickedPoint& point );

/// Checks that the picked point presents in the object's topology
[[deprecated( "use getPickedPointPosition() instead" )]] MRMESH_API MR_BIND_IGNORE bool isPickedPointValid( const VisualObject* object, const PickedPoint& point );

/// Returns object normal in local coordinates at given point,
/// returns std::nullopt if object or point is invalid, or if it is ObjectLines or ObjectPoints without normals
MRMESH_API std::optional<Vector3f> getPickedPointNormal( const VisualObject& object, const PickedPoint& point );

} //namespace MR
