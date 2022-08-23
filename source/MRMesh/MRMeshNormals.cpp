#include "MRMeshNormals.h"
#include "MRMesh.h"
#include "MRRingIterator.h"
#include "MRBuffer.h"
#include "MRVector4.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

FaceNormals computePerFaceNormals( const Mesh & mesh )
{
    MR_TIMER
    FaceId lastValidFace = mesh.topology.lastValidFace();

    const auto & edgePerFace = mesh.topology.edgePerFace();
    std::vector<Vector3f> res( lastValidFace + 1 );
    tbb::parallel_for( tbb::blocked_range<FaceId>( FaceId{0}, lastValidFace + 1 ), [&]( const tbb::blocked_range<FaceId> & range )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
        {
            auto e = edgePerFace[f];
            if ( !e.valid() )
                continue;
            res[f] = mesh.leftNormal( e );
        }
    } );

    return res;
}

Buffer<Vector4f> computePerFaceNormals4( const Mesh & mesh, size_t bufferSize )
{
    MR_TIMER
    FaceId lastValidFace = mesh.topology.lastValidFace();

    const auto & edgePerFace = mesh.topology.edgePerFace();
    assert( bufferSize >= lastValidFace + 1 );
    Buffer<Vector4f> res( bufferSize );
    tbb::parallel_for( tbb::blocked_range<FaceId>( FaceId{0}, lastValidFace + 1 ), [&]( const tbb::blocked_range<FaceId> & range )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
        {
            auto e = edgePerFace[f];
            if ( !e.valid() )
                continue;
            const auto norm = mesh.leftNormal( e );
            res[f] = Vector4f{ norm.x,norm.y,norm.z,1.0f };
        }
    } );

    return res;
}

VertexNormals computePerVertNormals( const Mesh & mesh )
{
    MR_TIMER
    VertId lastValidVert = mesh.topology.lastValidVert();

    std::vector<Vector3f> res( lastValidVert + 1 );
    tbb::parallel_for( tbb::blocked_range<VertId>( VertId{0}, lastValidVert + 1 ), [&]( const tbb::blocked_range<VertId> & range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            if ( mesh.topology.hasVert( v ) )
                res[v] = mesh.normal( v );
        }
    } );

    return res;
}

MeshNormals computeMeshNormals( const Mesh & mesh )
{
    MR_TIMER
    MeshNormals res;

    // compute directional areas of each mesh triangle
    FaceId lastValidFace = mesh.topology.lastValidFace();
    res.faceNormals.resize( lastValidFace + 1 );
    tbb::parallel_for( tbb::blocked_range<FaceId>( FaceId{0}, lastValidFace + 1 ), [&]( const tbb::blocked_range<FaceId> & range )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
        {
            if ( mesh.topology.hasFace( f ) )
                res.faceNormals[f] = mesh.dirDblArea( f );
        }
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

    VertId lastValidVert = mesh.topology.lastValidVert();
    res.vertNormals.resize( lastValidVert + 1 );
    tbb::parallel_for( tbb::blocked_range<VertId>( VertId{0}, lastValidVert + 1 ), [&]( const tbb::blocked_range<VertId> & range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            if ( mesh.topology.hasVert( v ) )
                res.vertNormals[v] = computeVertNormal( v );
        }
    } );

    // compute per-face normals from directional areas
    tbb::parallel_for( tbb::blocked_range<FaceId>( FaceId{0}, lastValidFace + 1 ), [&]( const tbb::blocked_range<FaceId> & range )
    {
        for ( FaceId f = range.begin(); f < range.end(); ++f )
        {
            res.faceNormals[f] = res.faceNormals[f].normalized();
        }
    } );

    return res;
}

Vector<TriangleCornerNormals, FaceId> computePerCornerNormals( const Mesh & mesh, const UndirectedEdgeBitSet * creases )
{
    MR_TIMER
    VertId lastValidVert = mesh.topology.lastValidVert();
    FaceId lastValidFace = mesh.topology.lastValidFace();

    Vector<TriangleCornerNormals, FaceId> res( lastValidFace + 1 );
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

    tbb::parallel_for( tbb::blocked_range<VertId>( VertId{0}, lastValidVert + 1 ), [&]( const tbb::blocked_range<VertId> & range )
    {
        for ( VertId v = range.begin(); v < range.end(); ++v )
        {
            auto e0 = mesh.topology.edgeWithOrg( v );
            if ( !e0 )
                continue;
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
        }
    } );

    return res;
}

} //namespace MR
