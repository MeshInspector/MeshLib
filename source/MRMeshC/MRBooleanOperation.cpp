#include "MRBooleanOperation.h"

#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRBitSet.h"
#include "MRMesh/MRBooleanOperation.h"

using namespace MR;

REGISTER_AUTO_CAST( BooleanResultMapper )
REGISTER_AUTO_CAST( EdgeBitSet )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( VertBitSet )
REGISTER_AUTO_CAST2( BooleanResultMapper::MapObject, MRBooleanResultMapperMapObject )
REGISTER_AUTO_CAST2( BooleanResultMapper::Maps, MRBooleanResultMapperMaps )
REGISTER_VECTOR( FaceMap )
REGISTER_VECTOR( VertMap )
REGISTER_VECTOR( WholeEdgeMap )

MRBooleanResultMapper* mrBooleanResultMapperNew( void )
{
    RETURN_NEW( BooleanResultMapper() );
}

MRFaceBitSet* mrBooleanResultMapperMapFaces( const MRBooleanResultMapper* mapper_, const MRFaceBitSet* oldBS_, MRBooleanResultMapperMapObject obj_ )
{
    ARG( mapper ); ARG( oldBS ); ARG_VAL( obj );
    RETURN_NEW( mapper.map( oldBS, obj ) );
}

MRVertBitSet* mrBooleanResultMapperMapVerts( const MRBooleanResultMapper* mapper_, const MRVertBitSet* oldBS_, MRBooleanResultMapperMapObject obj_ )
{
    ARG( mapper ); ARG( oldBS ); ARG_VAL( obj );
    RETURN_NEW( mapper.map( oldBS, obj ) );
}

MREdgeBitSet* mrBooleanResultMapperMapEdges( const MRBooleanResultMapper* mapper_, const MREdgeBitSet* oldBS_, MRBooleanResultMapperMapObject obj_ )
{
    ARG( mapper ); ARG( oldBS ); ARG_VAL( obj );
    RETURN_NEW( mapper.map( oldBS, obj ) );
}

MRFaceBitSet* mrBooleanResultMapperNewFaces( const MRBooleanResultMapper* mapper_ )
{
    ARG( mapper );
    RETURN_NEW( mapper.newFaces() );
}

MRFaceBitSet* mrBooleanResultMapperFilteredOldFaceBitSet( MRBooleanResultMapper* mapper_, const MRFaceBitSet* oldBS_, MRBooleanResultMapperMapObject obj_ )
{
    ARG( mapper ); ARG( oldBS ); ARG_VAL( obj );
    RETURN_NEW( mapper.filteredOldFaceBitSet( oldBS, obj ) );
}

const MRBooleanResultMapperMaps* mrBooleanResultMapperGetMaps( const MRBooleanResultMapper* mapper_, MRBooleanResultMapperMapObject index_ )
{
    ARG( mapper ); ARG_VAL( index );
    RETURN( &mapper.getMaps( index ) );
}

const MRFaceMap mrBooleanResultMapperMapsCut2origin( const MRBooleanResultMapperMaps* maps_ )
{
    ARG( maps );
    RETURN_VECTOR( maps.cut2origin );
}

const MRFaceMap mrBooleanResultMapperMapsCut2newFaces( const MRBooleanResultMapperMaps* maps_ )
{
    ARG( maps );
    RETURN_VECTOR( maps.cut2newFaces );
}

const MRWholeEdgeMap mrBooleanResultMapperMapsOld2newEdges( const MRBooleanResultMapperMaps* maps_ )
{
    ARG( maps );
    RETURN_VECTOR( maps.old2newEdges );
}

const MRVertMap mrBooleanResultMapperMapsOld2NewVerts( const MRBooleanResultMapperMaps* maps_ )
{
    ARG( maps );
    RETURN_VECTOR( maps.old2newVerts );
}

bool mrBooleanResultMapperMapsIdentity( const MRBooleanResultMapperMaps* maps_ )
{
    ARG( maps );
    return maps.identity;
}

void mrBooleanResultMapperFree( MRBooleanResultMapper* mapper_ )
{
    ARG_PTR( mapper );
    delete mapper;
}
