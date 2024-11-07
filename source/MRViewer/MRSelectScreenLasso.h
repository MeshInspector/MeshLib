#pragma once

#include "MRViewerFwd.h"
#include "MRMesh/MRVector2.h"

namespace MR
{

/**
    * calculate area on screen that is inside of closed contour given by screen points.
    * 
    * return the matrix of pixels (in local space of active viewport) belonging selected area
    */
MRVIEWER_API BitSet calculateSelectedPixelsInsidePolygon( const Contour2f & screenPoints );

/**
    * calculate area on screen that is near to open contour given by screen points.
    *
    * return the matrix of pixels (in local space of active viewport) belonging selected area
    */
MRVIEWER_API BitSet calculateSelectedPixelsNearPolygon( const Contour2f & screenPoints, float radiusPix );

/**
 * get faces ids of object located in selected area on viewport
 * @param pixBs the matrix of pixels (in local space of viewport) belonging selected area
 * @param onlyVisible get only visible faces (that no covered another faces)
 * @param includeBackfaces get also faces from back side of object
 */
MRVIEWER_API FaceBitSet findIncidentFaces( const Viewport& viewport, const BitSet& pixBs, const ObjectMesh& obj,
                                           bool onlyVisible = false, bool includeBackfaces = true, 
                                           const std::vector<ObjectMesh*> * occludingMeshes = nullptr ); // these meshes can influence face visibility in onlyVisible=true mode

/**
 * appends viewport visible faces (in pixBs) to visibleFaces
 * @param pixBs the matrix of pixels (in local space of viewport) belonging selected area
 * @param objects of interest
 * @param visibleFaces vector that correspond to objects and will be updated in this function
 * @param includeBackfaces get also faces from back side of object
 */
MRVIEWER_API void appendGPUVisibleFaces( const Viewport& viewport, const BitSet& pixBs, const std::vector<std::shared_ptr<ObjectMesh>>& objects,
    std::vector<FaceBitSet>& visibleFaces, bool includeBackfaces = true );

/**
 * get vertex ids of object located in selected area on viewport
 * @param bsVec the matrix of pixels (in local space of viewport) belonging to selected area
 * @param includeBackfaces get also vertices with normals not toward the camera
 * @param onlyVisible get only visible vertices (that no covered with clipping plane)
 */
MRVIEWER_API VertBitSet findVertsInViewportArea( const Viewport& viewport, const BitSet& bsVec, const ObjectPoints& obj,
                         bool includeBackfaces = true, bool onlyVisible = false );

}
