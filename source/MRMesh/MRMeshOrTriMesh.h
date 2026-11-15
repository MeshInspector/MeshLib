#pragma once

#include "MRMesh.h"
#include "MRTriMesh.h"
#include "MRSaveSettings.h"
#include "MRBitSet.h"

/// accessors allowing the same template code to work with both Mesh and TriMesh;
/// all vertices and triangles of TriMesh are always valid
namespace MR::detail
{

/// maps valid vertices to packed sequential indices (if packValidOnly = true) or keeps the same indices
inline VertRenumber getVertRenumber( const Mesh & mesh, bool packValidOnly )
{
    return VertRenumber( mesh.topology.getValidVerts(), packValidOnly );
}

/// identity vertex renumbering for TriMesh
struct NoRenumber
{
    int numVerts = 0;
    int sizeVerts() const { return numVerts; }
    int operator()( VertId v ) const { assert( v ); return (int)v; }
};

inline NoRenumber getVertRenumber( const TriMesh & mesh, bool )
{
    return { .numVerts = (int)mesh.points.size() };
}

inline VertId lastValidVert( const Mesh & mesh ) { return mesh.topology.lastValidVert(); }
inline VertId lastValidVert( const TriMesh & mesh ) { return VertId( (int)mesh.points.size() - 1 ); }

inline bool hasVert( const Mesh & mesh, VertId v ) { return mesh.topology.hasVert( v ); }
inline bool hasVert( const TriMesh &, VertId ) { return true; }

inline const VertBitSet & validVerts( const Mesh & mesh ) { return mesh.topology.getValidVerts(); }
inline VertBitSet validVerts( const TriMesh & mesh ) { return VertBitSet( mesh.points.size(), true ); }

inline FaceId lastValidFace( const Mesh & mesh ) { return mesh.topology.lastValidFace(); }
inline FaceId lastValidFace( const TriMesh & mesh ) { return FaceId( (int)mesh.tris.size() - 1 ); }

inline int numValidFaces( const Mesh & mesh ) { return mesh.topology.numValidFaces(); }
inline int numValidFaces( const TriMesh & mesh ) { return (int)mesh.tris.size(); }

inline bool hasFace( const Mesh & mesh, FaceId f ) { return mesh.topology.hasFace( f ); }
inline bool hasFace( const TriMesh &, FaceId ) { return true; }

inline const FaceBitSet & validFaces( const Mesh & mesh ) { return mesh.topology.getValidFaces(); }
inline FaceBitSet validFaces( const TriMesh & mesh ) { return FaceBitSet( mesh.tris.size(), true ); }

inline ThreeVertIds getTriVerts( const Mesh & mesh, FaceId f ) { return mesh.topology.getTriVerts( f ); }
inline const ThreeVertIds & getTriVerts( const TriMesh & mesh, FaceId f ) { return mesh.tris[f]; }

/// TriMesh with a region of triangles: only the triangles of the region and their vertices are valid
struct TriMeshPart
{
    const TriMesh & mesh;
    const VertCoords & points;
    const FaceBitSet & region;
    VertBitSet regionVerts;

    TriMeshPart( const TriMesh & m, const FaceBitSet & r ) : mesh( m ), points( m.points ), region( r ), regionVerts( m.points.size() )
    {
        assert( region.find_last() < (int)mesh.tris.size() );
        for ( FaceId f : region )
            for ( VertId v : mesh.tris[f] )
                regionVerts.set( v );
    }
};

inline VertRenumber getVertRenumber( const TriMeshPart & part, bool packValidOnly ) { return VertRenumber( part.regionVerts, packValidOnly ); }
inline VertId lastValidVert( const TriMeshPart & part ) { return part.regionVerts.find_last(); }
inline bool hasVert( const TriMeshPart & part, VertId v ) { return part.regionVerts.test( v ); }
inline const VertBitSet & validVerts( const TriMeshPart & part ) { return part.regionVerts; }
inline FaceId lastValidFace( const TriMeshPart & part ) { return part.region.find_last(); }
inline int numValidFaces( const TriMeshPart & part ) { return (int)part.region.count(); }
inline bool hasFace( const TriMeshPart & part, FaceId f ) { return part.region.test( f ); }
inline const FaceBitSet & validFaces( const TriMeshPart & part ) { return part.region; }
inline const ThreeVertIds & getTriVerts( const TriMeshPart & part, FaceId f ) { return part.mesh.tris[f]; }

template<typename MeshType>
Triangle3f getTriPoints( const MeshType & mesh, FaceId f )
{
    const auto & vs = getTriVerts( mesh, f );
    return { mesh.points[vs[0]], mesh.points[vs[1]], mesh.points[vs[2]] };
}

} //namespace MR::detail
