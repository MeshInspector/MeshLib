#include "MRVector.h"

#include "detail/TypeCast.h"

#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRVector3.h"

#include <vector>

using namespace MR;

REGISTER_AUTO_CAST( AffineXf3f )
REGISTER_AUTO_CAST( Vector3f )
REGISTER_AUTO_CAST2( std::vector<AffineXf3f>, MRVectorAffineXf3f )
REGISTER_AUTO_CAST2( std::vector<Vector3f>, MRVectorVector3f )

const MRAffineXf3f* mrVectorAffineXf3fData( const MRVectorAffineXf3f* vec_ )
{
    ARG( vec );
    RETURN( vec.data() );
}

size_t mrVectorAffineXf3fSize( const MRVectorAffineXf3f* vec_ )
{
    ARG( vec );
    return vec.size();
}

void mrVectorAffineXf3fFree( MRVectorAffineXf3f* vec_ )
{
    ARG_PTR( vec );
    delete vec;
}

const MRVector3f* mrVectorVector3fData( const MRVectorVector3f* vec_ )
{
    ARG( vec );
    RETURN( vec.data() );
}

size_t mrVectorVector3fSize( const MRVectorVector3f* vec_ )
{
    ARG( vec );
    return vec.size();
}

void mrVectorVector3fFree( MRVectorVector3f* vec_ )
{
    ARG_PTR( vec );
    delete vec;
}
