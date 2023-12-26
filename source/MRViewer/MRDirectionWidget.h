#pragma once

#include "exports.h"
#include "MRViewerEventsListener.h"
#include "MRViewer/MRViewer.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRChangeXfAction.h"

namespace MR
{

class MRVIEWER_CLASS DirectionWidget : public MultiListener<MouseDownListener, MouseMoveListener, MouseUpListener>
{
public:
    using OnDirectionChangedCallback = std::function<void( const Vector3f& )>;

private:
    std::shared_ptr<ObjectMesh> directionObj_;
    Vector3f dir_;
    Vector3f base_;
    float length_;
    bool mousePressed_ = false;
    Vector3f worldStartPoint_;
    float viewportStartPointZ_{ 0.0f };
    OnDirectionChangedCallback onDirectionChanged_;

    class ChangeDirAction : public ChangeXfAction
    {
    public:
        ChangeDirAction( DirectionWidget& plugin, const std::shared_ptr<Object>& obj ) :
            ChangeXfAction( "Change Dir", obj ),
            plugin_{ plugin },
            dir_{ plugin.dir_ }
        {}
        virtual void action( Type type ) override
        {
            ChangeXfAction::action( type );
            std::swap( dir_, plugin_.dir_ );

        }
    private:
        DirectionWidget& plugin_;
        Vector3f dir_;
    };

    void clear_();

public:

    MRVIEWER_API DirectionWidget( const Vector3f& dir, const Vector3f& base, float length, OnDirectionChangedCallback onDirectionChanged );
    MRVIEWER_API ~DirectionWidget();
    MRVIEWER_API void updateDirection( const Vector3f& dir );
    MRVIEWER_API void updateArrow( const Vector3f& base, float length );
    MRVIEWER_API void setVisible( bool visible );

private:
    MRVIEWER_API virtual bool onMouseDown_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseUp_( Viewer::MouseButton button, int modifier ) override;
    MRVIEWER_API virtual bool onMouseMove_( int mouse_x, int mouse_y ) override;
};

}