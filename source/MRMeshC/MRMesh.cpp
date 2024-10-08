#include "MRMesh.h"
#include "MRMeshTopology.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRBox.h"
#include "MRMesh/MRBuffer.h"
#include "MRMesh/MRMesh.h"

#include <span>

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( Box3f )
REGISTER_AUTO_CAST( EdgeId )
REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( MeshTopology )
REGISTER_AUTO_CAST( ThreeVertIds )
REGISTER_AUTO_CAST( Triangle3f )
REGISTER_AUTO_CAST( Vector3f )

MRMesh* mrMeshCopy( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN_NEW( mesh );
}

MRMesh* mrMeshFromTriangles( const MRVector3f* vertexCoordinates_, size_t vertexCoordinatesNum, const MRThreeVertIds* t_, size_t tNum )
{
    std::span vertexCoordinates { auto_cast( vertexCoordinates_ ), vertexCoordinatesNum };
    std::span t { auto_cast( t_ ), tNum };

    // TODO: cast instead of copying
    VertCoords vertexCoordinatesVec( vertexCoordinates.begin(), vertexCoordinates.end() );
    Triangulation tVec( t.begin(), t.end() );

    RETURN_NEW( Mesh::fromTriangles( vertexCoordinatesVec, tVec ) );
}

MRMesh* mrMeshFromTrianglesDuplicatingNonManifoldVertices( const MRVector3f* vertexCoordinates_, size_t vertexCoordinatesNum, const MRThreeVertIds* t_, size_t tNum )
{
    std::span vertexCoordinates { auto_cast( vertexCoordinates_ ), vertexCoordinatesNum };
    std::span t { auto_cast( t_ ), tNum };

    // TODO: cast instead of copying
    VertCoords vertexCoordinatesVec( vertexCoordinates.begin(), vertexCoordinates.end() );
    Triangulation tVec( t.begin(), t.end() );

    RETURN_NEW( Mesh::fromTrianglesDuplicatingNonManifoldVertices( vertexCoordinatesVec, tVec ) );
}

MRMesh* mrMeshNewFromPointTriples( const MRTriangle3f* posTriangles_, size_t posTrianglesNum, bool duplicateNonManifoldVertices )
{
    std::span posTriangles { auto_cast( posTriangles_ ), posTrianglesNum };

    // TODO: cast instead of copying
    std::vector<Triangle3f> posTrianglesVec( posTriangles.begin(), posTriangles.end() );

    RETURN_NEW( Mesh::fromPointTriples( posTrianglesVec, duplicateNonManifoldVertices ) );
}

const MRVector3f* mrMeshPoints( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN( mesh.points.data() );
}

MRVector3f* mrMeshPointsRef( MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN( mesh.points.data() );
}

size_t mrMeshPointsNum( const MRMesh* mesh_ )
{
    ARG( mesh );
    return mesh.points.size();
}

const MRMeshTopology* mrMeshTopology( const MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN( &mesh.topology );
}

MRMeshTopology* mrMeshTopologyRef( MRMesh* mesh_ )
{
    ARG( mesh );
    RETURN( &mesh.topology );
}

MRBox3f mrMeshComputeBoundingBox( const MRMesh* mesh_, const MRAffineXf3f* toWorld_ )
{
    ARG( mesh ); ARG_PTR( toWorld );
    RETURN( mesh.computeBoundingBox( toWorld ) );
}

void mrMeshTransform( MRMesh* mesh_, const MRAffineXf3f* xf_, const MRVertBitSet* region_ )
{
    ARG( mesh ); ARG( xf ); ARG_PTR_OF( VertBitSet, region );
    mesh.transform( xf, region );
}

void mrMeshAddPartByMask( MRMesh* mesh_, const MRMesh* from_, const MRFaceBitSet* fromFaces_, const MRMeshAddPartByMaskParameters* params )
{
    ARG( mesh ); ARG( from ); ARG_OF( FaceBitSet, fromFaces );

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

void mrMeshFree( MRMesh* mesh_ )
{
    ARG_PTR( mesh );
    delete mesh;
}

MRVector3f mrMeshHoleDirArea( const MRMesh* mesh_, MREdgeId e_ )
{
    ARG( mesh ); ARG_VAL( e );
    RETURN( (Vector3f)mesh.holeDirArea( e ) );
}

void mrMeshPack( MRMesh* mesh_, bool rearrangeTriangles )
{
    ARG( mesh );
    mesh.pack( nullptr, nullptr, nullptr, rearrangeTriangles );
}

void mrMeshPackOptimally( MRMesh* mesh_, bool preserveAABBTree )
{
    ARG( mesh );
    mesh.packOptimally( preserveAABBTree );
}

MRTriangulation* mrMeshGetTriangulation( const MRMesh* mesh )
{
    return mrMeshTopologyGetTriangulation( mrMeshTopology( mesh ) );
}

MREdgePath* mrMeshFindHoleRepresentiveEdges( const MRMesh* mesh )
{
    return mrMeshTopologyFindHoleRepresentiveEdges( mrMeshTopology( mesh ) );
}

double mrMeshVolume( const MRMesh* mesh_, const MRFaceBitSet* region_ )
{
    ARG( mesh ); ARG_PTR_OF( FaceBitSet, region );
    return mesh.volume( region );
}
