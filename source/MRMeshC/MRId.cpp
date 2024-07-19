#include "MRId.h"

#include "MRMesh/MRId.h"

using namespace MR;

static_assert( sizeof( MREdgeId ) == sizeof( EdgeId ) );
static_assert( sizeof( MRFaceId ) == sizeof( FaceId ) );
static_assert( sizeof( MRVertId ) == sizeof( VertId ) );
static_assert( sizeof( MRObjId ) == sizeof( ObjId ) );
