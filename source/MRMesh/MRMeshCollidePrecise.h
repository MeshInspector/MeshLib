#pragma once

#include "MRId.h"
#include "MRMeshPart.h"
#include <functional>

namespace MR
{

// edge from one mesh and triangle from another mesh
struct EdgeTri
{
    EdgeId edge;
    FaceId tri;
    EdgeTri() = default;
    EdgeTri( EdgeId e, FaceId t ) : edge( e ), tri( t ) { }
};

inline bool operator==( const EdgeTri& a, const EdgeTri& b )
{
    return a.edge.undirected() == b.edge.undirected() && a.tri == b.tri;
}

struct PreciseCollisionResult
{
    // each edge is directed to have its origin inside and its destination outside of the other mesh
    std::vector<EdgeTri> edgesAtrisB;
    std::vector<EdgeTri> edgesBtrisA;
};

// float-to-int coordinate converter
using ConvertToIntVector = std::function<Vector3i(const Vector3f&)>;
// int-to-float coordinate converter
using ConvertToFloatVector = std::function<Vector3f( const Vector3i& )>;
// this struct contains coordinate converters float-int-float
struct CoordinateConverters
{
    ConvertToIntVector toInt{};
    ConvertToFloatVector toFloat{};
};

// finds all pairs of colliding edges from one mesh and triangle from another mesh
MRMESH_API PreciseCollisionResult findCollidingEdgeTrisPrecise( const MeshPart & a, const MeshPart & b, 
    ConvertToIntVector conv, const AffineXf3f * rigidB2A = nullptr ); // rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation

// finds all intersections between every given edge from A and given triangles from B
MRMESH_API std::vector<EdgeTri> findCollidingEdgeTrisPrecise( 
    const Mesh & a, const std::vector<EdgeId> & edgesA,
    const Mesh & b, const std::vector<FaceId> & facesB,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A = nullptr );

// finds all intersections between every given triangle from A and given edge from B
MRMESH_API std::vector<EdgeTri> findCollidingEdgeTrisPrecise( 
    const Mesh & a, const std::vector<FaceId> & facesA,
    const Mesh & b, const std::vector<EdgeId> & edgesB,
    ConvertToIntVector conv, const AffineXf3f * rigidB2A = nullptr );

// creates converter from Vector3f to Vector3i in Box range (int diapason is mapped to box range)
MRMESH_API ConvertToIntVector getToIntConverter( const Box3d& box );
// creates converter from Vector3i to Vector3f in Box range (int diapason is mapped to box range)
MRMESH_API ConvertToFloatVector getToFloatConverter( const Box3d& box );

// creates simple converters from Vector3f to Vector3i and back in mesh parts area range
MRMESH_API CoordinateConverters getVectorConverters( const MeshPart& a, const MeshPart& b,
    const AffineXf3f* rigidB2A = nullptr );// rigid transformation from B-mesh space to A mesh space, nullptr considered as identity transformation


} //namespace MR
