#include "MRSaveSettings.h"
#include "MRVector.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

VertRenumber::VertRenumber( const VertBitSet & validVerts, bool saveValidOnly )
{
    MR_TIMER
    if ( saveValidOnly )
    {
        vert2packed_ = makeVectorWithSeqNums( validVerts );
        sizeVerts_ = (int)validVerts.count();
    }
    else
        sizeVerts_ = validVerts.find_last() + 1;
}

const VertCoords & transformPoints( const VertCoords & verts, const VertBitSet & validVerts, const AffineXf3d * xf, VertCoords & buf )
{
    if ( !xf )
        return verts;
    buf = verts;
    BitSetParallelFor( validVerts, [&]( VertId v )
    {
        buf[v] = Vector3f( (*xf)( Vector3d( buf[v] ) ) );
    } );
    return buf;
}

const VertNormals & transformNormals( const VertNormals & normals, const VertBitSet & validVerts, const Matrix3d * m, VertNormals & buf )
{
    if ( !m )
        return normals;
    buf = normals;
    BitSetParallelFor( validVerts, [&]( VertId v )
    {
        buf[v] = Vector3f( *m * Vector3d( buf[v] ) );
    } );
    return buf;
}

} //namespace MR
