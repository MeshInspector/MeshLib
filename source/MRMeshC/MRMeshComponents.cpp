#include "MRMeshComponents.h"
#include "MRBitSet.h"
#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRMeshComponents.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRBitSet.h"

#include <span>

using namespace MR;

MR_VECTOR_LIKE_IMPL( Face2RegionMap, RegionId )

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_AUTO_CAST( UndirectedEdgeBitSet )
REGISTER_VECTOR_LIKE( MRFace2RegionMap, RegionId )

namespace
{

MeshPart cast( MRMeshPart mp )
{
    return {
        *auto_cast( mp.mesh ),
        auto_cast( mp.region )
    };
}

} // namespace

MRFaceBitSet* mrMeshComponentsGetComponent( const MRMeshPart* mp, MRFaceId id, MRFaceIncidence incidence, const MRUndirectedEdgeBitSet* isCompBd_ )
{
    ARG_PTR( isCompBd );

    FaceBitSet result = MeshComponents::getComponent( cast( *mp ), FaceId(id.id), ( MeshComponents::FaceIncidence( incidence ) ), isCompBd );

    return mrFaceBitSetCopy( ( const MRFaceBitSet* )&result );
}

MRFaceBitSet* mrMeshComponentsGetLargestComponent( const MRMeshPart* mp, MRFaceIncidence incidence, const MRUndirectedEdgeBitSet* isCompBd_, float minArea, int* numSmallerComponents )
{
    ARG_PTR( isCompBd );

    FaceBitSet result = MeshComponents::getLargestComponent( cast( *mp ), ( MeshComponents::FaceIncidence( incidence ) ), isCompBd, minArea, numSmallerComponents );

    return mrFaceBitSetCopy( ( const MRFaceBitSet* )&result );
}

MRFaceBitSet* mrMeshComponentsGetLargeByAreaComponents( const MRMeshPart* mp, float minArea, const MRUndirectedEdgeBitSet* isCompBd_ )
{
    ARG_PTR( isCompBd );
    
    FaceBitSet result = MeshComponents::getLargeByAreaComponents( cast( *mp ), minArea, isCompBd );

    return mrFaceBitSetCopy( (const MRFaceBitSet* ) & result);
}

MRMeshComponentsMap mrMeshComponentsGetAllComponentsMap( const MRMeshPart* mp, MRFaceIncidence incidence )
{
    auto result = MeshComponents::getAllComponentsMap( cast( *mp ), ( MeshComponents::FaceIncidence( incidence ) ) );
    MRMeshComponentsMap ret;
    ret.numComponents = result.second;
    ret.faceMap = auto_cast( NEW_VECTOR(std::move(result.first.vec_ )));
    return ret;
}

MRMeshRegions mrMeshComponentsGetLargeByAreaRegions( const MRMeshPart* mp, const MRFace2RegionMap* face2RegionMap, int numRegions, float minArea )
{
    std::span<int> as{ (int*)face2RegionMap->data, face2RegionMap->size };
    // TODO: cast instead of copying
    MR::Vector<MR::RegionId, MR::FaceId> asVec( as.begin(), as.end() );
    auto result = MeshComponents::getLargeByAreaRegions( cast( *mp ), asVec, numRegions, minArea );
    return { .faces = auto_cast( new_from( result.first ) ), .numRegions = result.second};
}

void mrMeshComponentsAllComponentsMapFree( const MRMeshComponentsMap* map )
{
    if ( map->faceMap->data != NULL )
    {
       mrFace2RegionMapFree( map->faceMap );
    }
}