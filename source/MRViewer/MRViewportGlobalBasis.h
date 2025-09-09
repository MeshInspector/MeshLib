#pragma once
#include "MRViewerFwd.h"
#include "MRMesh/MRViewportId.h"
#include "MRMesh/MRObjectLines.h"
#include "MRMesh/MRSignal.h"

namespace MR
{

/// Class to unify Global Basis control
class MRVIEWER_CLASS ViewportGlobalBasis
{
public:
    /// Constructs default global basis object
    MRVIEWER_API ViewportGlobalBasis();

    /// Returns length of axis (all are the same)
    MRVIEWER_API float getAxesLength( ViewportId id = {} ) const;

    /// Returns width of axis (all are the same)
    MRVIEWER_API float getAxesWidth( ViewportId id = {} ) const;

    /// Sets length and width for all axes
    MRVIEWER_API void setAxesProps( float length, float width, ViewportId id = {} );

    /// Sets colors for each axis of this object
    MRVIEWER_API void setColors( const Color& xColor, const Color& yColor, const Color& zColor, const Color& labelColors );

    /// Simple accessor to visual children (useful for pickers or box calculations)
    MRVIEWER_API const std::vector<std::shared_ptr<VisualObject>>& axesChildren() const;

    /// returns true if any of its children requires redraw
    MRVIEWER_API bool getRedrawFlag( ViewportMask vpMask ) const;
    /// reset redraw flag for all children
    MRVIEWER_API void resetRedrawFlag() const;

    /// Draw this object into given viewport
    MRVIEWER_API void draw( const Viewport& vp ) const;

    /// Set visibility for all child objects
    MRVIEWER_API void setVisible( bool on, ViewportMask vpMask = ViewportMask::all() );

    /// returns true if object is present and visible
    bool isVisible( ViewportMask vpMask = ViewportMask::any() ) const { return !axes_.empty() && axes_[0] && axes_[0]->isVisible( vpMask ); }
    
    /// Set visibility for grid objects
    MRVIEWER_API void setGridVisible( bool on, ViewportMask vpMask = ViewportMask::all() );

    /// returns true if grid is present and visible
    bool isGridVisible( ViewportMask vpMask = ViewportMask::any() ) const { return !grids_.empty() && grids_[0] && grids_[0]->isVisible( vpMask ); }

    /// clears connections of this structure (by default it changes colors on theme change and change font size on rescale)
    void resetConnections() { connections_.clear(); }
private:
    std::vector<std::shared_ptr<MR::VisualObject>> axes_;
    std::vector<std::shared_ptr<ObjectLines>> grids_;
    std::vector<boost::signals2::scoped_connection> connections_;

    void creteGrids_();
    mutable ViewportProperty<Matrix3f> cachedGridRotation_;
    void updateGridXfs_( const Viewport& vp ) const;
};

}