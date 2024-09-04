#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"

MR_DOTNET_NAMESPACE_BEGIN

public value struct EdgeTri
{
    EdgeId edge;
    FaceId tri;
};

public ref class PreciseCollisionResult
{
public:
    ~PreciseCollisionResult();
    /// each edge is directed to have its origin inside and its destination outside of the other mesh
    property ReadOnlyCollection<EdgeTri>^ EdgesAtrisB { ReadOnlyCollection<EdgeTri>^ get(); }
    /// each edge is directed to have its origin inside and its destination outside of the other mesh
    property ReadOnlyCollection<EdgeTri>^ EdgesBtrisA { ReadOnlyCollection<EdgeTri>^ get(); }

internal:
    PreciseCollisionResult( MR::PreciseCollisionResult* nativeResult );

    MR::PreciseCollisionResult* getNativeResult() { return nativeResult_; }

private:
    List<EdgeTri>^ edgesAtrisB_;
    List<EdgeTri>^ edgesBtrisA_;

    MR::PreciseCollisionResult* nativeResult_;
};

public ref class MeshCollidePrecise
{
public:
    /**
    * \brief finds all pairs of colliding edges from one mesh and triangle from another mesh
    */
    static PreciseCollisionResult^ FindCollidingEdgeTrisPrecise( MeshPart meshA, MeshPart meshB, CoordinateConverters^ conv );
    /**
    * \brief finds all pairs of colliding edges from one mesh and triangle from another mesh
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    */
    static PreciseCollisionResult^ FindCollidingEdgeTrisPrecise( MeshPart meshA, MeshPart meshB, CoordinateConverters^ conv, AffineXf3f^ rigibB2A );
    /**
    * \brief finds all pairs of colliding edges from one mesh and triangle from another mesh
    * \param rigidB2A rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation
    * \param anyIntersection if true then the function returns as fast as it finds any intersection
    */
    static PreciseCollisionResult^ FindCollidingEdgeTrisPrecise( MeshPart meshA, MeshPart meshB, CoordinateConverters^ conv, AffineXf3f^ rigibB2A, bool anyInterssection );
};

MR_DOTNET_NAMESPACE_END

