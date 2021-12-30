#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"

namespace MR
{

Box3f PointCloud::getBoundingBox() const 
{ 
    return getAABBTree().getBoundingBox(); 
}

const AABBTreePoints& PointCloud::getAABBTree() const
{
    return AABBTreeOwner_.getOrCreate( [this]{ return AABBTreePoints( *this ); } );
}

} //namespace MR

