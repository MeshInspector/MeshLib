#include "MRVector.h"

#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"

#include <vector>

using namespace MR;

const MRAffineXf3f* mrVectorAffineXf3fData( const MRVectorAffineXf3f* vec_ )
{
    const auto& vec = *reinterpret_cast<const std::vector<AffineXf3f>*>( vec_ );
    return reinterpret_cast<const MRAffineXf3f*>( vec.data() );
}

size_t mrVectorAffineXf3fSize( const MRVectorAffineXf3f* vec_ )
{
    const auto& vec = *reinterpret_cast<const std::vector<AffineXf3f>*>( vec_ );
    return vec.size();
}

void mrVectorAffineXf3fFree( MRVectorAffineXf3f* vec )
{
    delete reinterpret_cast<std::vector<AffineXf3f>*>( vec );
}
