#pragma once
#include "MRMeshFwd.h"
#include "MRMesh.h"

#pragma managed( push, off )
#include <MRMesh/MRPrecisePredicates3.h>
#pragma managed( pop )

MR_DOTNET_NAMESPACE_BEGIN

public ref class CoordinateConverters
{
public:
    CoordinateConverters( MeshPart meshA, MeshPart meshB );
    ~CoordinateConverters();

internal:
    MR::ConvertToFloatVector* getConvertToFloatVector() { return convertToFloatVector_; }
    MR::ConvertToIntVector* getConvertToIntVector() { return convertToIntVector_; }

    MR::CoordinateConverters ToNative();
private:
    MR::ConvertToFloatVector* convertToFloatVector_;
    MR::ConvertToIntVector* convertToIntVector_;
};

MR_DOTNET_NAMESPACE_END