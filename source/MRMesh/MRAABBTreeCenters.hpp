#pragma once

#include "MRBox.h"

namespace MR
{

/// a triangle of a tree being built reduced to its index and the doubled center of its box: smaller to move and compared by one coordinate
struct TreeCenter
{
    int i; ///< the index of the triangle
    Vector3f c; ///< the doubled center of its box

    TreeCenter( int i, const Vector3f & c ) noexcept : i( i ), c( c ) { }
    explicit TreeCenter( NoInit ) noexcept : c( noInit ) { }
};
static_assert( sizeof( TreeCenter ) == 16 );

/// returns given box including the centers d[0..num); a NaN coordinate is skipped as by Box::include
[[nodiscard]] Box3f includeCenters( Box3f box, const TreeCenter * d, int num );

/// moves the center of order nth along the axis (ties are ordered by index) to its place, the smaller centers before it and the bigger ones after it,
/// as std::nth_element does
void selectCenter( TreeCenter * first, TreeCenter * nth, TreeCenter * last, int axis );

/// places the centers src[0..num) to dst[0..num) so that the first num / 2 of them are the smaller ones along the axis, as selectCenter does in place:
/// the centers are counted in buckets between lo and hi (their minimal and maximal coordinates), one pass moves the centers of the smaller buckets,
/// of the median's bucket and of the bigger buckets to their parts of dst, and only the median's bucket is left to select in
/// (or to split by its own buckets if it holds most of the centers); src is overwritten as scratch;
/// the counting and the moving are done by chunks of chunkSize centers in parallel; returns false and writes nothing if the buckets cannot be made
[[nodiscard]] bool splitCentersByBuckets( TreeCenter * src, TreeCenter * dst, int num, int axis, float lo, float hi, int chunkSize );

} //namespace MR
