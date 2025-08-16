#include "MRObjectMeshData.h"
#include "MRHeapBytes.h"
#include "MRColor.h"
#include "MRMesh.h"

namespace MR
{

ObjectMeshData ObjectMeshData::clone() const
{
    ObjectMeshData res = *this;
    if ( res.mesh )
        res.mesh = std::make_shared<Mesh>( *res.mesh );
    return res;
}

size_t ObjectMeshData::heapBytes() const
{
    return MR::heapBytes( mesh )
        + selectedFaces.heapBytes()
        + selectedEdges.heapBytes()
        + creases.heapBytes()
        + vertColors.heapBytes()
        + faceColors.heapBytes()
        + uvCoordinates.heapBytes()
        + texturePerFace.heapBytes();
}

} //namespace MR
