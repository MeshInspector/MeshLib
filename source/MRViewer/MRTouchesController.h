#pragma once
#include "MRViewerFwd.h"
#include "MRViewerEventsListener.h"
#include "MRMesh/MRVector2.h"
#include "MRMouse.h"
#include <vector>
#include <optional>

namespace MR
{
// class to operate with touches
// only overrides signals
class MRVIEWER_CLASS TouchesController : public MultiListener<TouchStartListener,TouchMoveListener,TouchEndListener>
{
public:
    MR_ADD_CTOR_DELETE_MOVE( TouchesController );
private:
    virtual bool onTouchStart_( int id, int x, int y ) override;
    virtual bool onTouchMove_( int id, int x, int y ) override;
    virtual bool onTouchEnd_( int id, int x, int y ) override;

    void setPos_( int id, int x, int y, bool on );

    std::vector<std::optional<Vector2i>> positions_;
    MouseButton mode_{MouseButton::Count}; // invalid value
    size_t secondTouchStartTime_{ 0 };
    float startDist_{0.0f};
    bool blockZoom_{false};
};

}