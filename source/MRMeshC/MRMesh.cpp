#include "MRMesh.h"

#include "MRMesh/MRMesh.h"

#include <span>

using namespace MR;

MRMesh* mrMeshCopy( const MRMesh* mesh )
{
    return reinterpret_cast<MRMesh*>( new Mesh( *reinterpret_cast<const Mesh*>( mesh ) ) );
}

MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates_, size_t vertexCoordinatesNum, const MRThreeVertIds* t_, size_t tNum )
{
    std::span vertexCoordinates { reinterpret_cast<const Vector3f*>( vertexCoordinates_ ), vertexCoordinatesNum };
    std::span t { reinterpret_cast<const ThreeVertIds*>( t_ ), tNum };

    // TODO: cast instead of copying
    VertCoords vertexCoordinatesVec( vertexCoordinates.begin(), vertexCoordinates.end() );
    Triangulation tVec( t.begin(), t.end() );

    auto* mesh = new Mesh( Mesh::fromTriangles( vertexCoordinatesVec, tVec ) );
    return reinterpret_cast<MRMesh*>( mesh );
}

MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates_, size_t vertexCoordinatesNum, const MRThreeVertIds* t_, size_t tNum )
{
    std::span vertexCoordinates { reinterpret_cast<const Vector3f*>( vertexCoordinates_ ), vertexCoordinatesNum };
    std::span t { reinterpret_cast<const ThreeVertIds*>( t_ ), tNum };

    // TODO: cast instead of copying
    VertCoords vertexCoordinatesVec( vertexCoordinates.begin(), vertexCoordinates.end() );
    Triangulation tVec( t.begin(), t.end() );

    auto* mesh = new Mesh( Mesh::fromTrianglesDuplicatingNonManifoldVertices( vertexCoordinatesVec, tVec ) );
    return reinterpret_cast<MRMesh*>( mesh );
}

MRMesh* mrMeshNewFromPointTriples( const MRTriangle3f* posTriangles_, size_t posTrianglesNum, bool duplicateNonManifoldVertices )
{
    std::span posTriangles { reinterpret_cast<const Triangle3f*>( posTriangles_ ), posTrianglesNum };

    // TODO: cast instead of copying
    std::vector<Triangle3f> posTrianglesVec( posTriangles.begin(), posTriangles.end() );

    auto* mesh = new Mesh( Mesh::fromPointTriples( posTrianglesVec, duplicateNonManifoldVertices ) );
    return reinterpret_cast<MRMesh*>( mesh );
}

const MRVector3f* mrMeshPoints( const MRMesh* mesh )
{
    return reinterpret_cast<const MRVector3f*>( reinterpret_cast<const Mesh*>( mesh )->points.data() );
}

MRVector3f* mrMeshPointsRef( MRMesh* mesh )
{
    return reinterpret_cast<MRVector3f*>( reinterpret_cast<Mesh*>( mesh )->points.data() );
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

void mrMeshTransform( MRMesh* mesh_, const MRAffineXf3f* xf_, const MRVertBitSet* region_ )
{
    auto& mesh = *reinterpret_cast<Mesh*>( mesh_ );
    const auto& xf = *reinterpret_cast<const AffineXf3f*>( xf_ );
    const auto* region = reinterpret_cast<const VertBitSet*>( region_ );

    mesh.transform( xf, region );
}

void mrMeshAddPartByMask( MRMesh* mesh_, const MRMesh* from_, const MRFaceBitSet* fromFaces_, const MRMeshAddPartByMaskParameters* params )
{
    auto& mesh = *reinterpret_cast<Mesh*>( mesh_ );
    const auto& from = *reinterpret_cast<const Mesh*>( from_ );
    const auto& fromFaces = *reinterpret_cast<const FaceBitSet*>( fromFaces_ );

    bool flipOrientation = false;
    // TODO: cast instead of copying
    std::vector<EdgePath> thisContoursVec;
    std::vector<EdgePath> fromContoursVec;
    // TODO: map
    if ( params )
    {
        flipOrientation = params->flipOrientation;
        std::span thisContours { reinterpret_cast<const EdgePath*>( params->thisContours ), params->thisContoursNum };
        std::span fromContours { reinterpret_cast<const EdgePath*>( params->fromContours ), params->fromContoursNum };
        thisContoursVec.assign( thisContours.begin(), thisContours.end() );
        fromContoursVec.assign( fromContours.begin(), fromContours.end() );
    }

    mesh.addPartByMask( from, fromFaces, flipOrientation, thisContoursVec, fromContoursVec );
}

void mrMeshFree( MRMesh* mesh )
{
    delete reinterpret_cast<Mesh*>( mesh );
}

MRVector3f mrMeshHoleDirArea( const MRMesh* mesh_, MREdgeId e_ )
{
    const auto& mesh = *reinterpret_cast<const Mesh*>( mesh_ );
    auto e = *reinterpret_cast<EdgeId*>( &e_ );

    const auto res = (Vector3f)mesh.holeDirArea( e );
    return reinterpret_cast<const MRVector3f&>( res );
}
