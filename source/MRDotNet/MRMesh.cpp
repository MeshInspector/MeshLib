#include "MRMesh.h"
#include "MRVector3.h"
#include "MRBox3.h"
#include "MRBitSet.h"
#include "MRAffineXf.h"
#include "MRMeshTriPoint.h"

#pragma managed( push, off )
#include <MRMesh/MRBuffer.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshSave.h>
#include <MRMesh/MRMeshTopology.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRRingIterator.h>

#include <MRMesh/MRCube.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRCylinder.h>

#pragma managed( pop )

#include <msclr/marshal_cppstd.h>

MR_DOTNET_NAMESPACE_BEGIN

Mesh::Mesh(MR::Mesh* mesh)
: mesh_( mesh )
{
    if ( !mesh_ )
        throw gcnew System::ArgumentNullException( "mesh" );
}

Mesh::~Mesh()
{
    delete mesh_;
}

VertCoordsReadOnly^ Mesh::Points::get( )
{
    if ( !points_ )
    {
        const auto& points = mesh_->points;
        points_ = gcnew VertCoords( int( points.size() ) );
        for ( size_t i = 0; i < points.size(); i++ )
            points_->Add( gcnew Vector3f( new MR::Vector3f( points.vec_[i] ) ) );
    }
    return points_->AsReadOnly();
}

Mesh^ Mesh::FromTriangles( VertCoords^ points, MR::DotNet::Triangulation^ triangles )
{
    if ( !points || !triangles )
        throw gcnew System::ArgumentNullException();

    MR::VertCoords nativePoints;
    nativePoints.reserve( points->Count );
    
    for each ( Vector3f^ point in points )
        nativePoints.push_back( *point->vec() );

    MR::Triangulation nativeTriangles;
    nativeTriangles.reserve( triangles->Count );
    
    for each ( ThreeVertIds^ triangle in triangles )
        nativeTriangles.vec_.push_back( MR::ThreeVertIds{ MR::VertId{ triangle->v0 }, MR::VertId{ triangle->v1 }, MR::VertId{ triangle->v2 } } );

    return gcnew Mesh( new MR::Mesh( std::move( MR::Mesh::fromTriangles( nativePoints, nativeTriangles ) ) ) );
}

Mesh^ Mesh::FromTrianglesDuplicatingNonManifoldVertices( VertCoords^ points, MR::DotNet::Triangulation^ triangles )
{
    if ( !points || !triangles )
        throw gcnew System::ArgumentNullException();

    MR::VertCoords nativePoints;
    nativePoints.reserve( points->Count );
    
    for each ( Vector3f^ point in points )
        nativePoints.push_back( *point->vec() );

    MR::Triangulation nativeTriangles;
    nativeTriangles.reserve( triangles->Count );
    
    for each ( ThreeVertIds^ triangle in triangles )
        nativeTriangles.vec_.push_back( MR::ThreeVertIds{ MR::VertId{ triangle->v0 }, MR::VertId{ triangle->v1 }, MR::VertId{ triangle->v2 } } );

    return gcnew Mesh( new MR::Mesh( std::move( MR::Mesh::fromTrianglesDuplicatingNonManifoldVertices( nativePoints, nativeTriangles ) ) ) );
}

Mesh^ Mesh::FromAnySupportedFormat( System::String^ path )
{
    if ( !path )
        throw gcnew System::ArgumentNullException();

    std::filesystem::path nativePath( msclr::interop::marshal_as<std::string>( path ) );
    auto meshOrErr = MR::MeshLoad::fromAnySupportedFormat( nativePath );

    if ( !meshOrErr )
        throw gcnew System::SystemException( gcnew System::String( meshOrErr.error().c_str() ) );
    
    return gcnew Mesh( new MR::Mesh( std::move( *meshOrErr ) ) );
}

void Mesh::ToAnySupportedFormat( Mesh^ mesh, System::String^ path )
{
    if ( !mesh )
        throw gcnew System::ArgumentNullException( "mesh" );

    if ( !path )
        throw gcnew System::ArgumentNullException( "path" );

    std::filesystem::path nativePath( msclr::interop::marshal_as<std::string>( path ) );
    auto err = MR::MeshSave::toAnySupportedFormat( *mesh->mesh_, nativePath );

    if ( !err )
    {
        std::string error = err.error();
        throw gcnew System::SystemException( gcnew System::String( error.c_str() ) );
    }
}

VertBitSetReadOnly^ Mesh::ValidPoints::get()
{
    if ( !validPoints_ )
        validPoints_ = gcnew VertBitSet( new MR::VertBitSet( mesh_->topology.getValidVerts() ) );

    return validPoints_;
}

FaceBitSetReadOnly^ Mesh::ValidFaces::get()
{
    if ( !validFaces_ )
        validFaces_ = gcnew FaceBitSet( new MR::FaceBitSet( mesh_->topology.getValidFaces() ) );

    return validFaces_;
}

TriangulationReadOnly^ Mesh::Triangulation::get()
{
    if ( !triangulation_ )
    {
        int size = int( mesh_->topology.faceSize() );
        triangulation_ = gcnew MR::DotNet::Triangulation( size );
        for ( size_t i = 0; i < size; i++ )
        {
            MR::ThreeVertIds threeVerts;
            mesh_->topology.getTriVerts( MR::FaceId( i ), threeVerts );
            triangulation_->Add( MR::DotNet::ThreeVertIds{ threeVerts[0], threeVerts[1], threeVerts[2] } );
        }
    }

    return triangulation_->AsReadOnly();
}

EdgePathReadOnly^ Mesh::HoleRepresentiveEdges::get()
{
    if ( !holeRepresentiveEdges_ )
    {
        auto nativePath = mesh_->topology.findHoleRepresentiveEdges();
        holeRepresentiveEdges_ = gcnew EdgePath( int( nativePath.size() ) );

        for ( MR::EdgeId e = MR::EdgeId{ 0 }; e < nativePath.size(); e++ )
        {
            holeRepresentiveEdges_->Add( EdgeId( e ) );
        }
    }
    return holeRepresentiveEdges_->AsReadOnly();
}

array<VertId>^ Mesh::GetLeftTriVerts( EdgeId e )
{
    MR::ThreeVertIds threeVerts;
    mesh_->topology.getLeftTriVerts( MR::EdgeId{ e }, threeVerts );
    return gcnew array<VertId>( 3 )
    {
        VertId( threeVerts[0] ),
        VertId( threeVerts[1] ),
        VertId( threeVerts[2] )
    };
}

Box3f^ Mesh::BoundingBox::get()
{
    if ( !boundingBox_ )
        boundingBox_ = gcnew Box3f( new MR::Box3f( std::move( mesh_->computeBoundingBox() ) ) );

    return boundingBox_;
}


bool Mesh::operator==( Mesh^ a, Mesh^ b )
{
    if ( !a || !b )
        throw gcnew System::ArgumentNullException();

    return *a->mesh_ == *b->mesh_;
}

bool Mesh::operator!=( Mesh^ a, Mesh^ b )
{
    if ( !a || !b )
        throw gcnew System::ArgumentNullException();

    return *a->mesh_ != *b->mesh_;
}

void Mesh::Transform( AffineXf3f^ xf )
{
    return mesh_->transform( *xf->xf() );
}

void Mesh::Transform( AffineXf3f^ xf, VertBitSet^ region )
{
    if ( !xf )
        throw gcnew System::ArgumentNullException( "xf" );

    if ( !region )
        throw gcnew System::ArgumentNullException( "region" );

    MR::VertBitSet nativeRegion( region->bitSet()->m_bits.begin(), region->bitSet()->m_bits.end() );
    return mesh_->transform( *xf->xf(), &nativeRegion );
}

void Mesh::PackOptimally()
{
    mesh_->packOptimally();
    clearManagedResources();
}

Mesh^ Mesh::MakeCube( Vector3f^ size, Vector3f^ base )
{ 
    if ( !size || !base )
        throw gcnew System::ArgumentNullException();

    return gcnew Mesh( new MR::Mesh( std::move( MR::makeCube( *size->vec(), *base->vec() ) ) ) );
}

Mesh^ Mesh::MakeSphere( float radius, int vertexCount )
{
    if ( vertexCount < 1 )
        throw gcnew System::ArgumentException( "vertexCount" );

    return gcnew Mesh( new MR::Mesh( std::move( MR::makeSphere( { .radius = radius, .numMeshVertices = vertexCount }))));
}

Mesh^ Mesh::MakeTorus( float primaryRadius, float secondaryRadius, int primaryResolution, int secondaryResolution )
{
    if ( primaryResolution < 1 )
        throw gcnew System::ArgumentException( "primaryResolution" );
    if ( secondaryResolution < 1 )
        throw gcnew System::ArgumentException( "secondaryResolution" );

    return gcnew Mesh( new MR::Mesh( std::move( MR::makeTorus( primaryRadius, secondaryRadius, primaryResolution, secondaryResolution ) ) ) );
}

Mesh^ Mesh::MakeCylinder( float radius, float length )
{
    return MakeCylinder( radius, radius, 0, 2.0f * float( System::Math::PI ), length, 16 );
}

Mesh^ Mesh::MakeCylinder( float radius, float startAngle, float arcSize, float length )
{
    return MakeCylinder( radius, radius, startAngle, arcSize, length, 16 );
}

Mesh^ Mesh::MakeCylinder( float radius, float startAngle, float arcSize, float length, int resolution )
{
    return MakeCylinder( radius, radius, startAngle, arcSize, length, resolution );
}

Mesh^ Mesh::MakeCylinder( float radius0, float radius1, float startAngle, float arcSize, float length, int resolution )
{
    return gcnew Mesh( new MR::Mesh( std::move( MR::makeCylinderAdvanced( radius0, radius1, startAngle, arcSize, length, resolution ) ) ) );
}

MeshProjectionResult Mesh::FindProjection( Vector3f^ point, MeshPart meshPart )
{
    return FindProjection( point, meshPart, FLT_MAX, nullptr, 0 );
}

MeshProjectionResult Mesh::FindProjection( Vector3f^ point, MeshPart meshPart, float maxDistanceSquared )
{
    return FindProjection( point, meshPart, maxDistanceSquared, nullptr, 0 );
}

MeshProjectionResult Mesh::FindProjection( Vector3f^ point, MeshPart meshPart, float maxDistanceSquared, AffineXf3f^ xf )
{
    return FindProjection( point, meshPart, maxDistanceSquared, xf, 0 );
}

MeshProjectionResult Mesh::FindProjection( Vector3f^ point, MeshPart meshPart, float maxDistanceSquared, AffineXf3f^ xf, float minDistanceSquared )
{
    if ( !point || !meshPart.mesh )
        throw gcnew System::ArgumentNullException();
    
    MR::FaceBitSet nativeRegion;
    if ( meshPart.region )
        nativeRegion = MR::FaceBitSet( meshPart.region->bitSet()->m_bits.begin(), meshPart.region->bitSet()->m_bits.end() );

    MR::MeshPart nativeMeshPart( *meshPart.mesh->getMesh(), meshPart.region ? &nativeRegion : nullptr );
    MR::Vector3f nativePoint( *point->vec() );
    MR::AffineXf3f nativeXf( xf ? *xf->xf() : MR::AffineXf3f() );

    auto nativeRes = MR::findProjection( nativePoint, nativeMeshPart,  maxDistanceSquared, xf? &nativeXf : nullptr, minDistanceSquared );

    MeshProjectionResult res;
    res.pointOnFace.faceId = nativeRes.proj.face;
    res.pointOnFace.point = gcnew Vector3f( new MR::Vector3f( nativeRes.proj.point ) );
    res.meshTriPoint = gcnew MeshTriPoint( new MR::MeshTriPoint( nativeRes.mtp ) );
    res.distanceSquared = nativeRes.distSq;

    return res;       
}

void Mesh::clearManagedResources()
{    
    points_ = nullptr; 
    triangulation_ = nullptr;    
    validPoints_ = nullptr;    
    validFaces_ = nullptr;
    holeRepresentiveEdges_ = nullptr;
}

MeshPart::MeshPart( Mesh^ m )
{
    if ( !m )
        throw gcnew System::ArgumentNullException( "mesh" );

    mesh = m;
}

MeshPart::MeshPart( Mesh^ m, FaceBitSet^ r )
{
    if ( !m )
        throw gcnew System::ArgumentNullException( "mesh" );

    mesh = m;
    region = r;
}

MR_DOTNET_NAMESPACE_END