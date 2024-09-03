#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"

#pragma managed( push, off )
#include <MRMesh/MRMeshCollidePrecise.h>
#pragma managed( pop )

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
    property ReadOnlyCollection<EdgeTri>^ EdgesAtrisB { ReadOnlyCollection<EdgeTri>^ get(); }
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
    static PreciseCollisionResult^ FindCollidingEdgeTrisPrecise( MeshPart meshA, MeshPart meshB, CoordinateConverters^ conv );
    static PreciseCollisionResult^ FindCollidingEdgeTrisPrecise( MeshPart meshA, MeshPart meshB, CoordinateConverters^ conv, AffineXf3f^ rigibB2A );
    static PreciseCollisionResult^ FindCollidingEdgeTrisPrecise( MeshPart meshA, MeshPart meshB, CoordinateConverters^ conv, AffineXf3f^ rigibB2A, bool anyInterssection );
};

MR_DOTNET_NAMESPACE_END

