#include "MRMesh.h"

#include "MRMesh/MRMesh.h"

#include <span>

using namespace MR;

MRMesh* mrMeshCopy( const MRMesh* mesh )
{
    return reinterpret_cast<MRMesh*>( new Mesh( *reinterpret_cast<const Mesh*>( mesh ) ) );
}

MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates_, size_t vertexCoordinatesNum, const int* t_, size_t tNum )
{
    std::span vertexCoordinates { reinterpret_cast<const Vector3f*>( vertexCoordinates_ ), vertexCoordinatesNum };
    assert( tNum % 3 == 0 );
    std::span t { reinterpret_cast<const ThreeVertIds*>( t_ ), tNum / 3 };

    // TODO: cast instead of copying
    VertCoords vertexCoordinatesVec( vertexCoordinates.begin(), vertexCoordinates.end() );
    Triangulation tVec( t.begin(), t.end() );

    auto* mesh = new Mesh( Mesh::fromTriangles( vertexCoordinatesVec, tVec ) );
    return reinterpret_cast<MRMesh*>( mesh );
}

MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates_, size_t vertexCoordinatesNum, const int* t_, size_t tNum )
{
    std::span vertexCoordinates { reinterpret_cast<const Vector3f*>( vertexCoordinates_ ), vertexCoordinatesNum };
    assert( tNum % 3 == 0 );
    std::span t { reinterpret_cast<const ThreeVertIds*>( t_ ), tNum / 3 };

    // TODO: cast instead of copying
    VertCoords vertexCoordinatesVec( vertexCoordinates.begin(), vertexCoordinates.end() );
    Triangulation tVec( t.begin(), t.end() );

    auto* mesh = new Mesh( Mesh::fromTrianglesDuplicatingNonManifoldVertices( vertexCoordinatesVec, tVec ) );
    return reinterpret_cast<MRMesh*>( mesh );
}

MRMesh* mrMeshNewFromPointTriples( const MRVector3f* posTriangles_, size_t posTrianglesNum, bool duplicateNonManifoldVertices )
{
    assert( posTrianglesNum % 3 == 0 );
    std::span posTriangles { reinterpret_cast<const Triangle3f*>( posTriangles_ ), posTrianglesNum / 3 };

    // TODO: cast instead of copying
    std::vector<Triangle3f> posTrianglesVec( posTriangles.begin(), posTriangles.end() );

    auto* mesh = new Mesh( Mesh::fromPointTriples( posTrianglesVec, duplicateNonManifoldVertices ) );
    return reinterpret_cast<MRMesh*>( mesh );
}

const MRVector3f* mrMeshPoints( const MRMesh* mesh )
{
    return reinterpret_cast<const MRVector3f*>( reinterpret_cast<const Mesh*>( mesh )->points.data() );
}

size_t mrMeshPointsNum( const MRMesh* mesh )
{
    return reinterpret_cast<const Mesh*>( mesh )->points.size();
}

const MRMeshTopology* mrMeshTopology( const MRMesh* mesh )
{
    return reinterpret_cast<const MRMeshTopology*>( &reinterpret_cast<const Mesh*>( mesh )->topology );
}

MRMeshTopology* mrMeshTopologyRef( MRMesh* mesh )
{
    return reinterpret_cast<MRMeshTopology*>( &reinterpret_cast<Mesh*>( mesh )->topology );
}

void mrMeshFree( MRMesh* mesh )
{
    delete reinterpret_cast<Mesh*>( mesh );
}
