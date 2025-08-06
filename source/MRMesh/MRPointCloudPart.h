#pragma once

#include "MRMeshFwd.h"

namespace MR
{

/// represents full point cloud (if region is nullptr) or some portion of point cloud (if region pointer is valid)
struct PointCloudPart
{
    const PointCloud& cloud;
    const VertBitSet* region = nullptr; // nullptr here means all valid points of point cloud

    PointCloudPart( const PointCloud& c, const VertBitSet* bs = nullptr ) noexcept : cloud( c ), region( bs )  {}

    // Make this assignable. A better idea would be to rewrite the class to not use references, but doing this instead preserves API compatibility.
    PointCloudPart( const PointCloudPart& other ) noexcept = default;
    PointCloudPart& operator=( const PointCloudPart& other ) noexcept
    {
        if ( this != &other )
        {
            // In modern C++ the result doesn't need to be `std::launder`ed, right?
            this->~PointCloudPart();
            ::new( ( void* )this ) PointCloudPart( other );
        }
        return *this;
    }
};

} // namespace MR
