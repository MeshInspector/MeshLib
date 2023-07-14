#include "MRCudaAccessor.h"
#include "MRMesh/MRPointsToMeshProjector.h"
#include "MRMesh/MRFastWindingNumber.h"

namespace MR
{

void CudaAccessor::setCudaAvailable( bool val )
{
    instance_().isCudaAvailable_ = val;
}

void CudaAccessor::setCudaFreeMemoryFunc( CudaFreeMemoryFunc freeMemFunc )
{
    instance_().freeMemFunc_ = freeMemFunc;
}

void CudaAccessor::setCudaFastWindingNumberConstructor( CudaFwnConstructor fwnCtor )
{
    instance_().fwnCtor_ = fwnCtor;
}

void CudaAccessor::setCudaMeshProjectorConstructor( CudaMeshProjectorConstructor mpCtor )
{
    instance_().mpCtor_ = mpCtor;
}

bool CudaAccessor::isCudaAvailable()
{
    auto& inst = instance_();
    return inst.isCudaAvailable_;
}

size_t CudaAccessor::getCudaFreeMemory()
{
    auto& inst = instance_();
    if ( !inst.freeMemFunc_ )
        return 0;
    return inst.freeMemFunc_();
}

std::unique_ptr<IFastWindingNumber> CudaAccessor::getCudaFastWindingNumber( const Mesh& mesh )
{
    auto& inst = instance_();
    if ( !inst.fwnCtor_ )
        return {};
    return inst.fwnCtor_( mesh );
}

std::unique_ptr<IPointsToMeshProjector> CudaAccessor::getCudaPointsToMeshProjector()
{
    auto& inst = instance_();
    if ( !inst.mpCtor_ )
        return {};
    return inst.mpCtor_();
}

CudaAccessor& CudaAccessor::instance_()
{
    static CudaAccessor instance;
    return instance;
}

}