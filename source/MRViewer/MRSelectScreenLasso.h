#pragma once

#include "MRViewerFwd.h"
#include "MRMesh/MRVector2.h"

namespace MR
{

/**
 * Class for selection area on screen
 */
class MRVIEWER_CLASS SelectScreenLasso
{
public:
    /// add point of area border
    MRVIEWER_API void addPoint( int mouseX, int mouseY );

    /// get current area border
    const std::vector<Vector2f>& getScreenLoop() { return screenLoop_; };
    
    /// clean area border
    void cleanScreenLoop() { screenLoop_.clear(); };

    /**
     * calculate area on screen that are inside for border.
     * closing border part is segment from begin to end point.
     * 
     * return the matrix of pixels (in local space of active viewport) belonging selected area
     */
    MRVIEWER_API BitSet calculateSelectedPixels( Viewer* viewer );

private:
    std::vector<Vector2f> screenLoop_;
};

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
 */
MRVIEWER_API VertBitSet findVertsInViewportArea( const Viewport& viewport, const BitSet& bsVec, const ObjectPoints& obj,
                         bool includeBackfaces = true );

}
