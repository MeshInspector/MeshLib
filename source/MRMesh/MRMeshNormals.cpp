#include "MRMeshNormals.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRBuffer.h"
#include "MRVector4.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

FaceNormals computePerFaceNormals( const Mesh & mesh )
{
    MR_TIMER;
    std::vector<Vector3f> res( mesh.topology.faceSize() );
    BitSetParallelFor( mesh.topology.getValidFaces(), [&]( FaceId f )
    {
        res[f] = mesh.normal( f );
    } );
    return res;
}

void computePerFaceNormals4( const Mesh & mesh, Vector4f* faceNormals, size_t size )
{
    MR_TIMER;
    size = std::min( size, mesh.topology.faceSize() );
    ParallelFor( 0_f, FaceId( size ), [&]( FaceId f )
    {
        if ( !mesh.topology.hasFace( f ) )
            return;
        const auto norm = mesh.normal( f );
        faceNormals[f] = Vector4f{ norm.x, norm.y, norm.z, 1.0f };
    } );
}

VertNormals computePerVertNormals( const Mesh & mesh )
{
    MR_TIMER;
    std::vector<Vector3f> res( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
    {
        res[v] = mesh.normal( v );
    } );
    return res;
}

VertNormals computePerVertPseudoNormals( const Mesh & mesh )
{
    MR_TIMER;
    std::vector<Vector3f> res( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
    {
        res[v] = mesh.pseudonormal( v );
    } );
    return res;
}

MeshNormals computeMeshNormals( const Mesh & mesh )
{
    MR_TIMER;
    MeshNormals res;

    // compute directional areas of each mesh triangle
    res.faceNormals.resize( mesh.topology.faceSize() );
    BitSetParallelFor( mesh.topology.getValidFaces(), [&]( FaceId f )
    {
        res.faceNormals[f] = mesh.dirDblArea( f );
    } );

    // compute per-vertex normals from directional areas
    auto computeVertNormal = [&]( VertId v )
    {
        Vector3f sum;
        for ( EdgeId e : orgRing( mesh.topology, v ) )
        {
            if ( auto f = mesh.topology.left( e ) )
            {
                sum += res.faceNormals[f];
            }
        }
        return sum.normalized();
    };

    res.vertNormals.resize( mesh.topology.vertSize() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
    {
        res.vertNormals[v] = computeVertNormal( v );
    } );

    // compute per-face normals from directional areas
    BitSetParallelFor( mesh.topology.getValidFaces(), [&]( FaceId f )
    {
        res.faceNormals[f] = res.faceNormals[f].normalized();
    } );

    return res;
}

Vector<TriangleCornerNormals, FaceId> computePerCornerNormals( const Mesh & mesh, const UndirectedEdgeBitSet * creases )
{
    MR_TIMER;

    Vector<TriangleCornerNormals, FaceId> res( mesh.topology.faceSize() );
    // converts edge to the normal in its left face at the corner of its origin
    auto edgeToLeftOrgNormal = [&]( EdgeId e ) -> Vector3f &
    {
        assert( e );
        auto f = mesh.topology.left( e );
        assert( f );
        int ne = 0;
        for ( EdgeId ei : leftRing( mesh.topology, f ) )
        {
            if ( ei == e )
                break;
            ++ne;
        }
        assert( ne < 3 );
        return res[f][ne];
    };

    BitSetParallelFor( mesh.topology.getValidVerts(), [&] ( VertId v )
    {
        auto e0 = mesh.topology.edgeWithOrg( v );
        assert( e0 );
        bool creaseVert = false;
        if ( creases )
        {
            for ( EdgeId e : orgRing( mesh.topology, e0 ) )
            {
                if ( creases->test( e.undirected() ) )
                {
                    creaseVert = true;
                    e0 = e;
                    break;
                }
            }
        }
        if ( !creaseVert )
        {
            const auto norm = mesh.normal( v );
            for ( EdgeId e : orgRing( mesh.topology, e0 ) )
                if ( mesh.topology.left( e ) )
                    edgeToLeftOrgNormal( e ) = norm;
        }
        else
        {
            // this vertex has at least one incident crease
            EdgeId efirst = e0;
            for ( ;;)
            {
                // compute average normal from crease edge to crease edge
                EdgeId elast = e0;
                Vector3f sum;
                for ( EdgeId e : orgRing( mesh.topology, efirst ) )
                {
                    if ( e != efirst && creases->test( e.undirected() ) )
                    {
                        elast = e;
                        break;
                    }
                    if ( mesh.topology.left( e ) )
                        sum += mesh.leftDirDblArea( e );
                }
                const auto norm = sum.normalized();
                for ( EdgeId e : orgRing( mesh.topology, efirst ) )
                {
                    if ( e != efirst && e == elast )
                        break;
                    if ( mesh.topology.left( e ) )
                        edgeToLeftOrgNormal( e ) = norm;
                }
                if ( elast == e0 )
                    break;
                efirst = elast;
            }
        }
    } );

    return res;
}

} //namespace MR
