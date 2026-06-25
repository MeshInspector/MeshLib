#pragma once

#if !( defined( MR_PARSING_FOR_ANY_BINDINGS ) || defined(MR_COMPILING_ANY_BINDINGS) )

#include "exports.h"
#include "MRCudaBasic.cuh"
#include "MRCudaMath.cuh"
#include "MRCudaPolyline.cuh"

#include "MRMesh/MRExpected.h"

namespace MR::Cuda
{

/// Helper class to manage the GPU memory-backed buffers for Polyline2 data
class Polyline2DataHolder
{
public:
    /// Allocates data buffers in the GPU memory and copies data to it.
    /// Returns error if the CUDA runtime couldn't allocate memory or copy host data.
    MRCUDA_API static Expected<Polyline2DataHolder> fromLines( const Polyline2& polyline );

    /// Returns data buffers.
    MRCUDA_API Polyline2Data data() const;
    operator Polyline2Data () const { return data(); }

    /// Resets data buffers.
    MRCUDA_API void reset();

    /// Computes the GPU memory amount required to allocate data for the polyline.
    MRCUDA_API static size_t heapBytes( const Polyline2& polyline );

private:
    DynamicArray<Node2> nodes_;
    DynamicArray<float2> points_;
    DynamicArray<int> orgs_;
};

/// Helper class to manage the GPU memory-backed buffers for Polyline3 data
class Polyline3DataHolder
{
public:
    /// Allocates data buffers in the GPU memory and copies data to it.
    /// Returns error if the CUDA runtime couldn't allocate memory or copy host data.
    MRCUDA_API static Expected<Polyline3DataHolder> fromLines( const Polyline3& polyline );

    /// Returns data buffers.
    MRCUDA_API Polyline3Data data() const;
    operator Polyline3Data () const { return data(); }

    /// Resets data buffers.
    MRCUDA_API void reset();

    /// Computes the GPU memory amount required to allocate data for the polyline.
    MRCUDA_API static size_t heapBytes( const Polyline3& polyline );

private:
    DynamicArray<Node3> nodes_;
    DynamicArray<float3> points_;
    DynamicArray<int> orgs_;
};

} // namespace MR::Cuda

#endif
