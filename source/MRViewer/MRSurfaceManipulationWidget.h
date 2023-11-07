#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRViewer.h"
#include "MRMesh/MRChangeMeshAction.h"

namespace MR
{

class MRVIEWER_CLASS SurfaceManipulationWidget // :
    //public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener,
    //                     KeyDownListener, KeyUpListener,
    //                     PreDrawListener>
{
public:
    struct Settings
    {
        float radius = 1.f; // [1 - ...]
        float force = 10.f; // [1 - 100]
        float intensity = 50.f; // [1 - 100]
    };

    MRVIEWER_API void init( const std::shared_ptr<ObjectMesh>& objectMesh );
    MRVIEWER_API void reset();

    MRVIEWER_API void setSettings( const Settings& settings );
    MRVIEWER_API const Settings& getSettings() { return settings_; };

    MRVIEWER_API bool onMouseDown_( Viewer::MouseButton button, int modifier );
    MRVIEWER_API bool onMouseUp_( Viewer::MouseButton button, int modifier );
    MRVIEWER_API bool onMouseMove_( int mouse_x, int mouse_y );

    MRVIEWER_API bool onKeyDown_( int key, int modifier );
    MRVIEWER_API bool onKeyUp_( int key, int modifier );

    MRVIEWER_API void preDraw_();
private:
    //MRVIEWER_API virtual bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    //MRVIEWER_API virtual bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    //MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;

    //MRVIEWER_API virtual bool onKeyDown_( int key, int modifier ) override;
    //MRVIEWER_API virtual bool onKeyUp_( int key, int modifier ) override;

    //MRVIEWER_API virtual void preDraw_() override;

    void changeSurface_();
    void updateUV_( bool set );
    void updateRegion_( const Vector2f& mousePos );

    bool active_ = false;
    Settings settings_;

    std::shared_ptr<ObjectMesh> obj_;
    float diagonal_ = 1.f;
    //VertId vert_;
    Vector2f mousePos_;
    bool mouseMoved_ = false;
    VertBitSet region_;
    VertBitSet regionExpanded_;
    VertScalars distances_;
    VertUVCoords uvs_;
    std::shared_ptr<ChangeMeshAction> changeMeshAction_;

    bool mousePressed_ = false;
    float direction_ = 1.f;
    bool onlySmooth_ = false;
};

}
