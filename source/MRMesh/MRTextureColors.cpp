#include "MRTextureColors.h"
#include "MRMesh.h"
#include "MRMeshTexture.h"
#include "MRColor.h"
#include "MRTimer.h"
#include "MRRingIterator.h"
#include "MRVector2.h"
#include "MRBitSetParallelFor.h"

namespace MR
{

static UVCoord getUV( VertId v, const ThreeVertIds& tri, const ThreeUVCoords& uvs )
{
    for ( int i = 0; i < 3; ++i )
    {
        if ( tri[i] != v )
            continue;
        return uvs[i];
    }
    assert( false ); // no such vertex in the triangle
    return {};
}

Color sampleVertexColor( const Mesh& mesh, VertId v, const MeshTexture& tex, const TriCornerUVCoords& triCornerUvCoords )
{
    Vector4f sumWC;
    float sumW = 0;
    for ( EdgeId e : orgRing( mesh.topology, v ) )
    {
        auto l = mesh.topology.left( e );
        if ( !l )
            continue;
        //auto lvs = mesh.topology.getTriVerts( l );
        const auto d0 = edgeVector( mesh.topology, mesh.points, e );
        const auto d1 = edgeVector( mesh.topology, mesh.points, mesh.topology.next( e ) );
        const auto angle = MR::angle( d0, d1 );
        sumWC += angle * Vector4f( tex.sample( tex.filter, getUV( v, mesh.topology.getTriVerts( l ), triCornerUvCoords[l] ) ) );
        sumW += angle;
    }
    if ( sumW > 0 )
        return Color( sumWC/ sumW );
    return {};
}

VertColors sampleVertexColors( const Mesh& mesh, const MeshTexture& tex, const TriCornerUVCoords& triCornerUvCoords )
{
    MR_TIMER;
    VertColors res;
    res.resizeNoInit( mesh.points.size() );
    BitSetParallelFor( mesh.topology.getValidVerts(), [&]( VertId v )
    {
        res[v] = sampleVertexColor( mesh, v, tex, triCornerUvCoords );
    } );
    return res;
}

} //namespace MR
