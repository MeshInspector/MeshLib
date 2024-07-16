#pragma once
#include "MRMeshFwd.h"

MR_DOTNET_NAMESPACE_BEGIN

/// enumeration of all possible boolean operations
public enum class BooleanOperation
{
    InsideA,
    InsideB,
    OutsideA,
    OutsideB,
    Union,
    Intersection,
    DifferenceBA,
    DifferenceAB,
    Count
};

/// optional parameters for boolean operations
public value struct BooleanParameters
{
    /// transform from mesh `B` space to mesh `A` space
    AffineXf3f^ rigidB2A;
    /// if set merge all non-intersecting components
    bool mergeAllNonIntersectingComponents;
};

/// output of boolean operation
public value struct BooleanResult
{
    /// resulting mesh
    Mesh^ mesh;
};


public ref class MeshBoolean
{
public:
    /** \brief Performs CSG operation on two meshes
    *
    * \ingroup BooleanGroup
    * Makes new mesh - result of boolean operation on mesh `A` and mesh `B`
    * \param meshA Input mesh `A`
    * \param meshB Input mesh `B`
    * \param op CSG operation to perform
    *
    * \note Input meshes should have no self-intersections in intersecting zone
    * \note If meshes are not closed in intersecting zone some boolean operations are not allowed (as far as input meshes interior and exterior cannot be determined)
    */
    static BooleanResult Boolean( Mesh^ meshA, Mesh^ meshB, BooleanOperation op );
    /** \brief Performs CSG operation on two meshes
    *
    * \ingroup BooleanGroup
    * Makes new mesh - result of boolean operation on mesh `A` and mesh `B`
    * \param meshA Input mesh `A`
    * \param meshB Input mesh `B`
    * \param op CSG operation to perform
    * \param params optional parameters
    *
    * \note Input meshes should have no self-intersections in intersecting zone
    * \note If meshes are not closed in intersecting zone some boolean operations are not allowed (as far as input meshes interior and exterior cannot be determined)
    */
    static BooleanResult Boolean( Mesh^ meshA, Mesh^ meshB, BooleanOperation op, BooleanParameters params );
};

MR_DOTNET_NAMESPACE_END