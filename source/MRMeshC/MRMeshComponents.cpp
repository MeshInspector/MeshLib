#include "MRMeshComponents.h"
#include "detail/TypeCast.h"
#include "detail/Vector.h"

#include "MRMesh/MRMeshComponents.h"
#include "MRMesh/MRId.h"
#include "MRMesh/MRMeshPart.h"
#include "MRMesh/MRBitSet.h"

#include <span>

using namespace MR;

REGISTER_AUTO_CAST( Mesh )
REGISTER_AUTO_CAST( FaceBitSet )
REGISTER_VECTOR( Face2RegionMap )

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

MRMeshComponentsMap mrMeshComponentsGetAllComponentsMap( const MRMeshPart* mp, MRFaceIncidence incidence )
{
    auto result = MeshComponents::getAllComponentsMap( cast( *mp ), ( MeshComponents::FaceIncidence( incidence ) ) );
    MRMeshComponentsMap ret;
    ret.numComponents = result.second;
    ret.faceMap = new MRFace2RegionMap();
    ret.faceMap->size = result.first.size();
    ret.faceMap->data = new MRRegionId[ret.faceMap->size];
    std::copy( result.first.vec_.begin(), result.first.vec_.end(), (RegionId*)ret.faceMap->data );
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
        delete[] map->faceMap->data;
        map->faceMap->data = nullptr;
        map->faceMap->size = 0;
    }
}