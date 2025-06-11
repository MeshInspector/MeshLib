#include "MRVector.h"

#include "detail/Vector.h"

#include "MRMesh/MRAffineXf.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRMatrix3.h"
#include "MRMesh/MRVector3.h"

using namespace MR;

MR_VECTOR_IMPL( AffineXf3f )
MR_VECTOR_IMPL( Vector3f )

MR_VECTOR_LIKE_IMPL( FaceMap, FaceId )
MR_VECTOR_LIKE_IMPL( WholeEdgeMap, EdgeId )
MR_VECTOR_LIKE_IMPL( VertMap, VertId )

MR_VECTOR_LIKE_IMPL( Scalars, float )
